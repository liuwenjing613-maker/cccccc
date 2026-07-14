import torch
import torch.nn as nn

class CausalDilatedConv1d(nn.Module):
    """
    Causal 1D Convolution with Dilation.
    
    Padding is dynamically calculated and applied strictly to the past frames 
    to ensure causality (preventing any future data leakage). This is essential 
    for real-time Active Noise Control applications.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        
        # Calculate padding required to maintain temporal alignment
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        # Crop the padded future frames to maintain strict causality
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return self.prelu(x)

class ResidualBlock(nn.Module):
    """
    1D-CNN Residual Block.
    
    Utilizes causal dilated convolutions and a 1x1 convolution for channel 
    mixing. The residual skip-connection enhances gradient flow, allowing 
    for deeper time-domain network architectures.
    """
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.dilated_conv = CausalDilatedConv1d(channels, channels, kernel_size, dilation)
        self.conv_1x1 = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        residual = x
        out = self.dilated_conv(x)
        out = self.conv_1x1(out)
        return out + residual

class TimeDomainANC(nn.Module):
    """
    Ultra-low latency Time-Domain 1D-CNN for Active Noise Control.
    
    Operates directly on raw audio waveforms point-by-point. Achieves a massive 
    receptive field through exponentially increasing dilation rates, eliminating 
    the need for STFT framing buffers and subsequent algorithmic delays.
    """
    def __init__(self, in_channels=1, out_channels=1, hidden_channels=32, num_layers=10):
        super().__init__()
        # Initial feature extraction layer
        self.input_conv = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        
        self.res_blocks = nn.ModuleList()
        # Exponentially increasing dilation: 1, 2, 4, 8, 16...
        for i in range(num_layers):
            dilation = 2 ** i
            self.res_blocks.append(
                ResidualBlock(hidden_channels, kernel_size=3, dilation=dilation)
            )
            
        # Output projection layers
        self.output_conv1 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1)
        self.prelu = nn.PReLU()
        self.output_conv2 = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Ensure input has the correct channel dimension: [Batch, Channels, Time]
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        out = self.input_conv(x)
        
        # Pass through the dilated residual blocks to capture long-term dependencies
        for block in self.res_blocks:
            out = block(out)
            
        out = self.prelu(self.output_conv1(out))
        out = self.output_conv2(out)
        
        # Remove the channel dimension for subsequent physical path processing: [Batch, Time]
        return out.squeeze(1)