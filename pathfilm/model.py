import torch
import torch.nn as nn


class CausalDilatedConv1d(nn.Module):
    """
    Causal 1D convolution with dilation.

    Same baseline idea:
    use padding, then crop the right side to avoid future leakage.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return self.prelu(x)


class PathEncoder(nn.Module):
    """
    Encode secondary path sh into a compact path embedding.

    Input:
        sh: [B, L]

    Output:
        path_emb: [B, emb_dim]

    Design:
        1. Small encoder to reduce path-ID overfitting.
        2. RMS-normalize sh, but keep log RMS as explicit feature.
        3. Adaptive pooling makes it robust to path length.
    """
    def __init__(self, emb_dim=16):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=15, stride=4, padding=7),
            nn.PReLU(),
            nn.Conv1d(8, 16, kernel_size=15, stride=4, padding=7),
            nn.PReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.fc = nn.Sequential(
            nn.Linear(16 + 1, emb_dim),
            nn.PReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, sh):
        if sh.dim() == 1:
            sh = sh.unsqueeze(0)

        eps = 1e-8

        rms = torch.sqrt(torch.mean(sh ** 2, dim=1, keepdim=True) + eps)
        log_rms = torch.log(rms + eps)

        sh_norm = sh / (rms + eps)
        feat = self.conv(sh_norm.unsqueeze(1)).squeeze(-1)

        feat = torch.cat([feat, log_rms], dim=1)
        path_emb = self.fc(feat)

        return path_emb


class PathFiLMResidualBlock(nn.Module):
    """
    Residual block with weak Path-FiLM modulation.

    Baseline:
        out = ConvBlock(x)

    E2:
        out = ConvBlock(x)
        out = out * (1 + gamma(sh)) + beta(sh)

    gamma and beta are bounded by tanh and film_scale.
    The FiLM layer is zero-initialized, so training starts from baseline behavior.
    """
    def __init__(
        self,
        channels,
        kernel_size,
        dilation,
        path_emb_dim=16,
        film_scale=0.05,
    ):
        super().__init__()

        self.dilated_conv = CausalDilatedConv1d(
            channels,
            channels,
            kernel_size,
            dilation,
        )

        self.conv_1x1 = nn.Conv1d(channels, channels, kernel_size=1)

        self.film_scale = film_scale
        self.film = nn.Linear(path_emb_dim, channels * 2)

        # Zero init:
        # gamma = 0, beta = 0 at start.
        # So the block initially behaves like the baseline residual block.
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, path_emb):
        residual = x

        out = self.dilated_conv(x)
        out = self.conv_1x1(out)

        gamma_beta = self.film(path_emb)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=1)

        gamma = self.film_scale * torch.tanh(gamma).unsqueeze(-1)
        beta = self.film_scale * torch.tanh(beta).unsqueeze(-1)

        out = out * (1.0 + gamma) + beta

        return out + residual


class TimeDomainANC(nn.Module):
    """
    E2 Path-FiLM ANC model.

    Compared with sample2048 baseline:
        - same causal time-domain CNN backbone
        - add secondary-path encoder
        - add weak FiLM modulation
        - no CausalGate
        - no attention

    forward:
        x:  [B, T]
        sh: [B, L]
    """
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        num_layers=10,
        path_emb_dim=16,
        path_dropout=0.35,
        film_scale=0.05,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.path_emb_dim = path_emb_dim
        self.path_dropout = path_dropout

        self.path_encoder = PathEncoder(emb_dim=path_emb_dim)

        self.input_conv = nn.Conv1d(
            in_channels,
            hidden_channels,
            kernel_size=1,
        )

        self.res_blocks = nn.ModuleList()

        for i in range(num_layers):
            dilation = 2 ** i
            self.res_blocks.append(
                PathFiLMResidualBlock(
                    channels=hidden_channels,
                    kernel_size=3,
                    dilation=dilation,
                    path_emb_dim=path_emb_dim,
                    film_scale=film_scale,
                )
            )

        self.output_conv1 = nn.Conv1d(
            hidden_channels,
            hidden_channels,
            kernel_size=1,
        )
        self.prelu = nn.PReLU()
        self.output_conv2 = nn.Conv1d(
            hidden_channels,
            out_channels,
            kernel_size=1,
        )

    def _make_path_embedding(self, sh, batch_size, device):
        if sh is None:
            return torch.zeros(batch_size, self.path_emb_dim, device=device)

        path_emb = self.path_encoder(sh)

        if self.training and self.path_dropout > 0:
            keep_mask = (
                torch.rand(batch_size, 1, device=path_emb.device)
                > self.path_dropout
            ).float()
            path_emb = path_emb * keep_mask

        return path_emb

    def forward(self, x, sh=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        batch_size = x.shape[0]
        device = x.device

        path_emb = self._make_path_embedding(
            sh=sh,
            batch_size=batch_size,
            device=device,
        )

        out = self.input_conv(x)

        for block in self.res_blocks:
            out = block(out, path_emb)

        out = self.prelu(self.output_conv1(out))
        out = self.output_conv2(out)

        return out.squeeze(1)