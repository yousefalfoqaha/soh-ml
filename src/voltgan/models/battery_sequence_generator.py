import torch
from torch import nn


def _group_norm(channels: int, num_groups: int = 8) -> nn.GroupNorm:
    return nn.GroupNorm(min(num_groups, channels), channels)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 2,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        self.norm = _group_norm(out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ConvTransposeBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 2,
        is_last: bool = False,
    ):
        super().__init__()

        self.conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            output_padding=1,
        )

        self.is_last = is_last

        if not is_last:
            self.norm = _group_norm(out_channels)
            self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)

        if self.is_last:
            return x

        return self.act(self.norm(x))


class ConvEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        base_channels: int,
        channel_mults: list[int],
        kernel_size: int = 7,
    ):
        super().__init__()
        channels = [in_channels] + [base_channels * m for m in channel_mults]

        blocks: list[nn.Module] = []
        for i in range(len(channel_mults)):
            blocks.append(
                ConvBlock(channels[i], channels[i + 1], kernel_size, stride=2)
            )

        blocks.append(nn.Conv1d(channels[-1], 2 * latent_dim, kernel_size=1))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        mu, logvar = h.chunk(2, dim=1)
        logvar = logvar.clamp(-8.0, 8.0)

        return mu, logvar


class _FiLM(nn.Module):
    def __init__(self, n_conditions: int, embed_dim: int):
        super().__init__()
        self.cond_scale = nn.Linear(n_conditions, embed_dim)
        self.cond_shift = nn.Linear(n_conditions, embed_dim)
        nn.init.zeros_(self.cond_scale.weight)
        nn.init.zeros_(self.cond_scale.bias)
        nn.init.zeros_(self.cond_shift.weight)
        nn.init.zeros_(self.cond_shift.bias)

    def forward(self, x: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        scale = self.cond_scale(conditions).unsqueeze(-1)
        shift = self.cond_shift(conditions).unsqueeze(-1)

        return x * (1.0 + scale) + shift


class ConvDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        out_channels: int,
        base_channels: int,
        channel_mults: list[int],
        n_conditions: int,
        kernel_size: int = 7,
    ):
        super().__init__()
        rev_mults = list(reversed(channel_mults))
        channels = [base_channels * m for m in rev_mults]

        blocks: list[nn.Module] = [nn.Conv1d(latent_dim, channels[0], kernel_size=1)]

        for i in range(len(rev_mults)):
            in_ch = channels[i]
            out_ch = channels[i + 1] if i + 1 < len(channels) else out_channels
            is_last = i == len(rev_mults) - 1
            blocks.append(
                ConvTransposeBlock(
                    in_ch, out_ch, kernel_size, stride=2, is_last=is_last
                )
            )

        self.net = nn.Sequential(*blocks)
        self.film = _FiLM(n_conditions=n_conditions, embed_dim=out_channels)

    def forward(
        self,
        z: torch.Tensor,
        conditions: torch.Tensor,
    ) -> torch.Tensor:
        out = self.net(z)

        return self.film(out, conditions)


class BatterySequenceGenerator(nn.Module):
    def __init__(
        self,
        n_conditions: int,
        padded_length: int,
        latent_length: int,
        latent_dim: int,
        conv_base_channels: int,
        conv_channel_mults: list[int],
        conv_kernel_size: int = 7,
    ):
        super().__init__()
        self.padded_length = padded_length
        self.latent_length = latent_length
        self.latent_dim = latent_dim
        self.n_conditions = n_conditions

        self.encoder = ConvEncoder(
            in_channels=3,
            latent_dim=latent_dim,
            base_channels=conv_base_channels,
            channel_mults=conv_channel_mults,
            kernel_size=conv_kernel_size,
        )

        self.decoder = ConvDecoder(
            latent_dim=latent_dim,
            out_channels=2,
            base_channels=conv_base_channels,
            channel_mults=conv_channel_mults,
            n_conditions=n_conditions,
            kernel_size=conv_kernel_size,
        )

    def encode(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(y.permute(0, 2, 1))

        return mu.permute(0, 2, 1), logvar.permute(0, 2, 1)

    def decode(
        self,
        z: torch.Tensor,
        conditions: torch.Tensor,
    ) -> torch.Tensor:
        out = self.decoder(z.permute(0, 2, 1), conditions)
        out = out.permute(0, 2, 1)

        return out[:, : self.padded_length, :]

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            codings_std = torch.exp(0.5 * logvar)
            noise = torch.randn_like(codings_std)

            return mu + noise * codings_std

        return mu

    def forward(
        self,
        y: torch.Tensor,
        conditions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(y)
        z = self.reparameterize(mu, logvar)
        y_hat = self.decode(z, conditions)

        return y_hat, mu, logvar
