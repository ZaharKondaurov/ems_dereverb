import torch
from torch import nn, Tensor
from src.causal_convs import CausalConv1d, CausalConvTranspose1d
from einops import rearrange


class TRA(nn.Module):
    """Temporal Recurrent Attention"""
    def __init__(self, channels):
        super().__init__()
        self.att_gru = nn.GRU(channels, channels*2, 1, batch_first=True)
        self.att_fc = nn.Linear(channels*2, channels)
        self.att_act = nn.Sigmoid()

    def forward(self, x):
        """x: (B,C,T,F)"""
        zt = torch.mean(x.pow(2), dim=-1)  # (B,C,T)
        # print(zt.shape)
        at = self.att_gru(zt)[0]
        at = self.att_fc(at)
        at = self.att_act(at)
        At = at[..., None]  # (B,C,T,1)

        return x * At
    

class TRAEncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bs: int = 1,
                 conv: nn.Module = nn.Conv1d, normalize: bool = True, is_sub: bool = False, deconv: bool = False):
        super().__init__()

        conv_module = nn.ConvTranspose1d if deconv else nn.Conv1d
        # print(in_channels // 2)
        self.conv1 = conv_module(in_channels=in_channels // 2, out_channels=in_channels // 2 * 3,
                              kernel_size=3, stride=1, padding=1, padding_mode="zeros")
        
        self.conv2 = conv_module(in_channels=in_channels // 2 * 3, out_channels=in_channels // 2 * 3,
                        kernel_size=3, stride=1, padding=1, padding_mode="zeros")
        
        self.conv3 = conv_module(in_channels=in_channels // 2 * 3, out_channels=in_channels // 2,
                        kernel_size=3, stride=1, padding=1, padding_mode="zeros")

        if normalize:
            self.norm1 = nn.BatchNorm1d(num_features=in_channels // 2 * 3)
            self.norm2 = nn.BatchNorm1d(num_features=in_channels // 2 * 3)
            self.norm3 = nn.BatchNorm1d(num_features=in_channels // 2)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
            self.norm3 = nn.Identity()

        if is_sub:
            self.activate1 = nn.ReLU()
            self.activate2 = nn.ReLU()
            self.activate3 = nn.ReLU()
        else:
            self.activate1 = nn.ELU()
            self.activate2 = nn.ELU()
            self.activate3 = nn.ELU()
        # self.activate = nn.ELU()
        # print(in_channels // 2)
        self.tra = TRA(in_channels // 2)
        # self.bs = bs

    def shuffle(self, x1, x2):
        """x1, x2: (B,C,F)"""
        x = torch.stack([x1, x2], dim=1)
        x = x.transpose(1, 2).contiguous()  # (B,C,2,F)
        x = rearrange(x, 'b c g f -> b (c g) f')  # (B,2C,F)
        return x

    def forward(self, complex_spectrum: Tensor, batch: int):
        """
        :param complex_spectrum: (batch * frames, channels, frequency)
        :return:
        """
        x1, x2 = torch.chunk(complex_spectrum, chunks=2, dim=1)
        # print(x1.shape)
        h1 = self.activate1(self.norm1(self.conv1(x1)))
        h1 = self.activate2(self.norm2(self.conv2(h1)))
        h1 = self.activate3(self.norm3(self.conv3(h1)))

        x1 = x1 + h1

        _, ch, freq = x1.shape
        # print(_, ch, freq)
        x1 = torch.reshape(x1, (batch, -1, ch, freq)) # rearrange(x1, '(b t) c f -> b t c f')

        x1 = self.tra(x1)

        x1 = torch.reshape(x1, (-1, ch, freq))

        complex_spectrum = self.shuffle(x1, x2)

        return complex_spectrum


class FullBandEncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int,
                 conv: nn.Module = nn.Conv1d, normalize: bool = True, is_sub: bool = False, act: nn.Module = nn.ELU):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, padding_mode="zeros")

        if normalize:
            self.norm = nn.BatchNorm1d(num_features=out_channels)
        else:
            self.norm = nn.Identity()

        if is_sub:
            self.activate = act() # nn.ReLU()
        else:
            self.activate = act() # nn.ELU()
        # self.activate = nn.ELU()

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
        # print(complex_spectrum.size())
        return complex_spectrum


class FullBandDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, output_padding: int=0,
                 conv: nn.Module = nn.Conv1d, conv_transposed: nn.Module = nn.ConvTranspose1d, mag_act: nn.Module = nn.ReLU, pha_act: nn.Module = nn.ELU, normalize: bool = True, split_act: bool = False, is_sub: bool = False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=in_channels // 2,
                              kernel_size=1, stride=1, padding=0)
        self.convT = nn.ConvTranspose1d(in_channels // 2, out_channels, kernel_size=kernel_size, stride=stride,
                                        padding=padding, output_padding=output_padding)

        if normalize:
            self.norm = nn.BatchNorm1d(num_features=out_channels)
        else:
            self.norm = nn.Identity()

        self.split_act = split_act

        if split_act:
            self.act1 = mag_act() # nn.Sigmoid()# nn.ReLU() # mag_act() # nn.ReLU()
            self.act2 = pha_act() # nn.ELU() # nn.Tanh()
        else:
            if is_sub:
                self.activate = mag_act() # nn.Sigmoid() # nn.ReLU() # mag_act() # nn.ReLU()
            else:
                self.activate = pha_act()

    def forward(self, encode_complex_spectrum: Tensor, decode_complex_spectrum):
        """
        :param decode_complex_spectrum: (batch * frames, channels1, frequency)
        :param encode_complex_spectrum: (batch * frames, channels2, frequency)
        :return:
        """
        # print("decode", encode_complex_spectrum.size(), decode_complex_spectrum.size())
        complex_spectrum = torch.cat([encode_complex_spectrum, decode_complex_spectrum], dim=1)
        complex_spectrum = self.conv(complex_spectrum)
        # print("decode conv out:", complex_spectrum.size())
        complex_spectrum = self.convT(complex_spectrum)
        # print("decode convT out:", complex_spectrum.size())
        complex_spectrum = self.norm(complex_spectrum)
        if not self.split_act:
            complex_spectrum = self.activate(complex_spectrum)
        else:
            _, channels, _ = complex_spectrum.shape
            abs_part, phase_part = complex_spectrum[:, :channels//2, :], complex_spectrum[:, channels//2:, :]
            abs_part = self.act1(abs_part)
            phase_part = self.act2(phase_part) # * torch.pi
            # print(abs_part.shape, phase_part.shape)
            complex_spectrum = torch.concat([abs_part, phase_part], dim=1)

        return complex_spectrum


class SubBandEncoderBlock(nn.Module):
    def __init__(self, start_frequency: int,
                 end_frequency: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 conv: nn.Module = nn.Conv1d,
                 mag_pha: nn.Module = nn.ELU):
        super().__init__()
        self.start_frequency = start_frequency
        self.end_frequency = end_frequency

        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.activate = mag_pha() # nn.ELU() # nn.ReLU()

    def forward(self, amplitude_spectrum: Tensor):
        """
        :param amplitude_spectrum: (batch*frames, channels, frequency)
        :return:
        """
        # print(f"Input [{self.start_frequency, self.end_frequency}]:", amplitude_spectrum.shape)
        sub_spectrum = amplitude_spectrum[:, :, self.start_frequency:self.end_frequency]
        # print(f"Sub spectrum [{self.start_frequency, self.end_frequency}]:", sub_spectrum.shape)
        # print("encoder in:", sub_spectrum.shape)
        sub_spectrum = self.conv(sub_spectrum)  # (batch*frames, out_channels, sub_bands)
        if torch.isnan(sub_spectrum).any().item() is True:
            print(f"[{(self.start_frequency, self.end_frequency)}] Conv has NaNs: ", torch.isnan(sub_spectrum).any().item())
        # print("encoder out:", sub_spectrum.shape)
        sub_spectrum = self.activate(sub_spectrum)
        if torch.isnan(sub_spectrum).any().item() is True:
            print(f"[{(self.start_frequency, self.end_frequency)}] After activation: ", torch.isnan(sub_spectrum).any().item())
        # print(f"Output [{self.start_frequency, self.end_frequency}]:", sub_spectrum.shape)
        return sub_spectrum
    

class SubBandEncoderBlock_baseline(nn.Module):
    def __init__(self, start_frequency: int,
                 end_frequency: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 conv: nn.Module = nn.Conv1d,):
        super().__init__()
        self.start_frequency = start_frequency
        self.end_frequency = end_frequency

        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.activate = nn.ReLU()

    def forward(self, amplitude_spectrum: Tensor):
        """
        :param amplitude_spectrum: (batch*frames, channels, frequency)
        :return:
        """
        # print(f"Input [{self.start_frequency, self.end_frequency}]:", amplitude_spectrum.shape)
        sub_spectrum = amplitude_spectrum[:, :, self.start_frequency:self.end_frequency]
        # print(f"Sub spectrum [{self.start_frequency, self.end_frequency}]:", sub_spectrum.shape)
        # print("encoder in:", sub_spectrum.shape)
        sub_spectrum = self.conv(sub_spectrum)  # (batch*frames, out_channels, sub_bands)
        if torch.isnan(sub_spectrum).any().item() is True:
            print(f"[{(self.start_frequency, self.end_frequency)}] Conv has NaNs: ", torch.isnan(sub_spectrum).any().item())
        # print("encoder out:", sub_spectrum.shape)
        sub_spectrum = self.activate(sub_spectrum)
        if torch.isnan(sub_spectrum).any().item() is True:
            print(f"[{(self.start_frequency, self.end_frequency)}] After activation: ", torch.isnan(sub_spectrum).any().item())
        # print(f"Output [{self.start_frequency, self.end_frequency}]:", sub_spectrum.shape)
        return sub_spectrum


# class SubBandDecoderBlock(nn.Module):
#     def __init__(self, in_features: int, out_features: int, start_idx: int, end_idx: int):
#         super().__init__()
#         self.start_idx = start_idx
#         self.end_idx = end_idx
#         self.fc = nn.Linear(in_features=in_features, out_features=out_features)
#         self.activate = nn.ReLU()

#     def forward(self, encode_amplitude_spectrum: Tensor, decode_amplitude_spectrum: Tensor):
#         """

#         :param encode_amplitude_spectrum: (batch * frames, channels, sub_bands)
#         :param decode_amplitude_spectrum: (batch * frames, channels, sub_bands)
#         :return:
#         """
#         encode_amplitude_spectrum = encode_amplitude_spectrum[:, :, self.start_idx: self.end_idx]
#         print("SubBand decoder input: ", encode_amplitude_spectrum.shape, decode_amplitude_spectrum.shape)
#         spectrum = torch.cat([encode_amplitude_spectrum, decode_amplitude_spectrum], dim=1)  # channels cat
#         spectrum = torch.transpose(spectrum, dim0=1, dim1=2).contiguous()   # (*, bands, channels)
#         print("SubBand decoder input: ", spectrum.shape)
#         spectrum = self.fc(spectrum)  # (*, bands, band-width)
#         spectrum = self.activate(spectrum)
#         first_dim, bands, band_width = spectrum.shape
#         spectrum = torch.reshape(spectrum, shape=(first_dim, bands*band_width))
#         print("Decoder spectrum: ", spectrum.shape)
#         return spectrum

class SubBandDecoderBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, start_idx: int, end_idx: int, mag_act: nn.Module = nn.ReLU):
        super().__init__()
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.activate = mag_act() # nn.ReLU()

    def forward(self, encode_amplitude_spectrum: Tensor, decode_amplitude_spectrum: Tensor):
        """

        :param encode_amplitude_spectrum: (batch * frames, channels, sub_bands)
        :param decode_amplitude_spectrum: (batch * frames, channels, sub_bands)
        :return:
        """
        encode_amplitude_spectrum = encode_amplitude_spectrum[:, :, self.start_idx: self.end_idx]
        # print("SubBand decoder input: ", encode_amplitude_spectrum.shape, decode_amplitude_spectrum.shape)
        spectrum = torch.cat([encode_amplitude_spectrum, decode_amplitude_spectrum], dim=1)  # channels cat
        spectrum = torch.transpose(spectrum, dim0=1, dim1=2).contiguous()   # (*, bands, channels)
        # bands, channels = spectrum.shape[-2:]
        # spectrum = spectrum.reshape((*spectrum.shape[:-2], -1))
        # print("SubBand decoder input: ", spectrum.shape)
        spectrum = self.fc(spectrum)  # (*, bands, band-width)
        spectrum = self.activate(spectrum)
        first_dim, bands, band_width = spectrum.shape
        spectrum = torch.reshape(spectrum, shape=(first_dim, bands*band_width))
        # print("Decoder spectrum: ", spectrum.shape)
        return spectrum
