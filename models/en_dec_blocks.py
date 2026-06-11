import torch
from torch import nn, Tensor


class FullBandEncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        normalize: bool = True,
        act: nn.Module = nn.ELU,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode="zeros",
        )

        if normalize:
            self.norm = nn.BatchNorm1d(num_features=out_channels)
        else:
            self.norm = nn.Identity()

        self.activate = act()

    def forward(self, complex_spectrum: Tensor):
        """
        :param complex_spectrum: (batch * frames, channels, frequency)
        :return:
        """
        complex_spectrum = self.conv(complex_spectrum)
        if torch.isnan(complex_spectrum).any().item() is True:
            print(f"Conv has NaNs: ", torch.isnan(complex_spectrum).any().item())
        complex_spectrum = self.norm(complex_spectrum)
        if torch.isnan(complex_spectrum).any().item() is True:
            print(f"After norm: ", torch.isnan(complex_spectrum).any().item())
        complex_spectrum = self.activate(complex_spectrum)
        if torch.isnan(complex_spectrum).any().item() is True:
            print(f"After activation: ", torch.isnan(complex_spectrum).any().item())
        return complex_spectrum


class FullBandDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_padding: int = 0,
        mag_act: nn.Module = nn.ReLU,
        pha_act: nn.Module = nn.ELU,
        normalize: bool = True,
        split_act: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.convT = nn.ConvTranspose1d(
            in_channels // 2,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

        if normalize:
            self.norm = nn.BatchNorm1d(num_features=out_channels)
        else:
            self.norm = nn.Identity()

        self.split_act = split_act

        if split_act:
            self.act1 = mag_act()
            self.act2 = pha_act()
        else:
            self.activate = mag_act()

    def forward(self, encode_complex_spectrum: Tensor, decode_complex_spectrum):
        """
        :param decode_complex_spectrum: (batch * frames, channels1, frequency)
        :param encode_complex_spectrum: (batch * frames, channels2, frequency)
        :return:
        """
        complex_spectrum = torch.cat(
            [encode_complex_spectrum, decode_complex_spectrum], dim=1
        )
        complex_spectrum = self.conv(complex_spectrum)

        complex_spectrum = self.convT(complex_spectrum)

        complex_spectrum = self.norm(complex_spectrum)
        if not self.split_act:
            complex_spectrum = self.activate(complex_spectrum)
        else:
            _, channels, _ = complex_spectrum.shape
            abs_part, phase_part = (
                complex_spectrum[:, : channels // 2, :],
                complex_spectrum[:, channels // 2 :, :],
            )
            abs_part = self.act1(abs_part)
            phase_part = self.act2(phase_part)

            complex_spectrum = torch.concat([abs_part, phase_part], dim=1)

        return complex_spectrum


class SubBandEncoderBlock(nn.Module):
    def __init__(
        self,
        start_frequency: int,
        end_frequency: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        mag_act: nn.Module = nn.ELU,
    ):
        super().__init__()
        self.start_frequency = start_frequency
        self.end_frequency = end_frequency

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.activate = mag_act()

    def forward(self, amplitude_spectrum: Tensor):
        """
        :param amplitude_spectrum: (batch*frames, channels, frequency)
        :return:
        """
        sub_spectrum = amplitude_spectrum[
            :, :, self.start_frequency : self.end_frequency
        ]

        sub_spectrum = self.conv(
            sub_spectrum
        )  # (batch*frames, out_channels, sub_bands)
        if torch.isnan(sub_spectrum).any().item() is True:
            print(
                f"[{(self.start_frequency, self.end_frequency)}] Conv has NaNs: ",
                torch.isnan(sub_spectrum).any().item(),
            )

        sub_spectrum = self.activate(sub_spectrum)
        if torch.isnan(sub_spectrum).any().item() is True:
            print(
                f"[{(self.start_frequency, self.end_frequency)}] After activation: ",
                torch.isnan(sub_spectrum).any().item(),
            )
        return sub_spectrum


class SubBandDecoderBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        start_idx: int,
        end_idx: int,
        mag_act: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.activate = mag_act()

    def forward(
        self, encode_amplitude_spectrum: Tensor, decode_amplitude_spectrum: Tensor
    ):
        """

        :param encode_amplitude_spectrum: (batch * frames, channels, sub_bands)
        :param decode_amplitude_spectrum: (batch * frames, channels, sub_bands)
        :return:
        """
        encode_amplitude_spectrum = encode_amplitude_spectrum[
            :, :, self.start_idx : self.end_idx
        ]

        spectrum = torch.cat(
            [encode_amplitude_spectrum, decode_amplitude_spectrum], dim=1
        )
        spectrum = torch.transpose(
            spectrum, dim0=1, dim1=2
        ).contiguous()  # (*, bands, channels)

        spectrum = self.fc(spectrum)  # (*, bands, band-width)
        spectrum = self.activate(spectrum)
        first_dim, bands, band_width = spectrum.shape
        spectrum = torch.reshape(spectrum, shape=(first_dim, bands * band_width))
        return spectrum
