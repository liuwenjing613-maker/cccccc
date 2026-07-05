import torch
import torch.nn as nn


class CausalDilatedConv1d(nn.Module):
    """
    Causal 1D convolution with dilation.
    Same idea as the official baseline:
    use padding, then crop the right side to avoid future leakage.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
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
    Encode the secondary path sh into a compact path embedding.

    Input:
        sh: [B, L]

    Output:
        path_emb: [B, emb_dim]

    Design notes:
    1. Use a very small encoder to reduce path-ID overfitting.
    2. Normalize sh by RMS but keep log RMS as one explicit feature.
    3. Adaptive pooling makes it insensitive to exact path length.
    """

    def __init__(self, emb_dim=32):
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


class CausalFeatureGate(nn.Module):
    """
    Lightweight causal channel attention.

    This is NOT global SE attention.
    It does not pool over the whole time axis, so it does not look into the future.

    Since the input hidden feature is produced by causal convolutions,
    gate[:, :, t] depends only on current/past information.

    The final layer is zero-initialized so the module starts as identity:
        output = x * (1 + 0)
    """

    def __init__(self, channels, reduction=4, gate_scale=0.1):
        super().__init__()

        mid_channels = max(4, channels // reduction)
        self.gate_scale = gate_scale

        self.net = nn.Sequential(
            nn.Conv1d(channels, mid_channels, kernel_size=1),
            nn.PReLU(),
            nn.Conv1d(mid_channels, channels, kernel_size=1),
        )

        # Identity initialization: delta = 0 at the beginning.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        delta = self.gate_scale * torch.tanh(self.net(x))
        return x * (1.0 + delta)


class PathFiLMResidualBlock(nn.Module):
    """
    Residual block with:
    1. causal dilated convolution,
    2. 1x1 channel mixing,
    3. optional causal feature gate,
    4. weak Path-FiLM modulation.

    FiLM:
        out = out * (1 + gamma) + beta

    gamma and beta are generated from secondary-path embedding.
    They are bounded by tanh and small film_scale to reduce overfitting.
    """

    def __init__(
        self,
        channels,
        kernel_size,
        dilation,
        path_emb_dim=32,
        use_attention=False,
        film_scale=0.1,
    ):
        super().__init__()

        self.dilated_conv = CausalDilatedConv1d(
            channels,
            channels,
            kernel_size,
            dilation,
        )
        self.conv_1x1 = nn.Conv1d(channels, channels, kernel_size=1)

        self.use_attention = use_attention
        if use_attention:
            self.attn = CausalFeatureGate(channels)
        else:
            self.attn = nn.Identity()

        self.film_scale = film_scale
        self.film = nn.Linear(path_emb_dim, channels * 2)

        # Identity initialization:
        # gamma = 0, beta = 0, so the initial block behaves like baseline.
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, path_emb):
        residual = x

        out = self.dilated_conv(x)
        out = self.conv_1x1(out)
        out = self.attn(out)

        gamma_beta = self.film(path_emb)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=1)

        gamma = self.film_scale * torch.tanh(gamma).unsqueeze(-1)
        beta = self.film_scale * torch.tanh(beta).unsqueeze(-1)

        out = out * (1.0 + gamma) + beta

        return out + residual


class TimeDomainANC(nn.Module):
    """
    Path-conditioned time-domain ANC model.

    Compatible with the official baseline training pipeline,
    but forward now accepts:
        x:  [B, T]
        sh: [B, L]

    If sh is None, path embedding is set to zero.
    This keeps the model callable as model(x), although model(x, sh) is recommended.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        num_layers=10,
        path_emb_dim=32,
        path_dropout=0.2,
        attention_start_layer=6,
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

            # First version: only use attention in deeper layers.
            # Earlier layers learn low-level causal acoustic filters.
            # Later layers use attention to refine high-level features.
            use_attention = i >= attention_start_layer

            self.res_blocks.append(
                PathFiLMResidualBlock(
                    channels=hidden_channels,
                    kernel_size=3,
                    dilation=dilation,
                    path_emb_dim=path_emb_dim,
                    use_attention=use_attention,
                    film_scale=0.1,
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

        # Path dropout:
        # During training, randomly remove path conditioning so the model
        # cannot simply memorize training paths.
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