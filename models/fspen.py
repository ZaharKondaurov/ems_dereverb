import torch

from torch import nn, Tensor
from models.en_dec_blocks import *
from models.sequence_modules import *
from src.fspen_configs import TrainConfig
from models.fullband_enc_dec import FullBandEncoder, FullBandDecoder
from models.subband_enc_dec import SubBandEncoder, SubBandDecoder, SubBandEncoder_ext, SubBandDecoder_ext

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


class FullSubPathExtension(nn.Module):
    def __init__(self, configs: TrainConfig, full_band_encoder: nn.Module = FullBandEncoder,  
                 sub_band_encoder: nn.Module = SubBandEncoder, full_band_decoder: nn.Module = FullBandDecoder,
                 sub_band_decoder: nn.Module = SubBandDecoder, need_mask: bool = True, print_mask: bool = False):
        super().__init__()
        self.full_band_encoder = full_band_encoder(configs)
        self.sub_band_encoder = sub_band_encoder(configs)

        self.num_rnn_modules = configs.dual_path_extension["num_modules"]

        merge_split = configs.merge_split
        merge_channels = merge_split["channels"]
        merge_bands = merge_split["bands"]
        compress_rate = merge_split["compress_rate"]

        self.feature_merge_layer = nn.Sequential(
            nn.Linear(in_features=merge_channels, out_features=merge_channels//compress_rate),
            nn.ELU(),
            nn.Conv1d(in_channels=merge_bands, out_channels=merge_bands//compress_rate, kernel_size=1, stride=1)
        )

        self.dual_path_extension_rnn_list = nn.ModuleList()
        for _ in range(configs.dual_path_extension["num_modules"]):
            self.dual_path_extension_rnn_list.append(DualPathExtensionRNN(**configs.dual_path_extension["parameters"]))

        self.feature_split_layer = nn.Sequential(
            nn.Conv1d(in_channels=merge_bands//compress_rate, out_channels=merge_bands, kernel_size=1, stride=1),
            nn.Linear(in_features=merge_channels//compress_rate, out_features=merge_channels),
            nn.ELU()
        )

        self.full_band_decoder = full_band_decoder(configs)
        self.sub_band_decoder = sub_band_decoder(configs)

        self.mask_padding = nn.ConstantPad2d(padding=(1, 0, 0, 0), value=0.0)
        self.need_mask = need_mask
        self.print_mask = print_mask

    
    def encode(self, in_complex_spectrum: Tensor, in_amplitude_spectrum: Tensor):
        batch, frames, channels, frequency = in_complex_spectrum.shape

        complex_spectrum = torch.reshape(in_complex_spectrum, shape=(batch * frames, channels, frequency))
        amplitude_spectrum = torch.reshape(in_amplitude_spectrum, shape=(batch*frames, 1, frequency))

        full_band_encode_outs, global_feature = self.full_band_encoder(complex_spectrum)

        sub_band_encode_outs, local_feature = self.sub_band_encoder(amplitude_spectrum)

        merge_feature = torch.cat(tensors=[global_feature, local_feature], dim=2)  # feature cat
        merge_feature = self.feature_merge_layer(merge_feature)

        # (batch*frames, channels, frequency) -> (batch*frames, channels//2, frequency//2)
        _, channels, frequency = merge_feature.shape
        merge_feature = torch.reshape(merge_feature, shape=(batch, frames, channels, frequency))
        merge_feature = torch.permute(merge_feature, dims=(0, 3, 1, 2)).contiguous()

        return merge_feature, full_band_encode_outs, sub_band_encode_outs
    
    def decode(self, merge_feature: Tensor, full_band_encode_outs: Tensor, sub_band_encode_outs: Tensor):
        merge_feature = torch.permute(merge_feature, dims=(0, 2, 3, 1)).contiguous()
        batch, frames, channels, frequency = merge_feature.shape
        merge_feature = torch.reshape(merge_feature, shape=(batch * frames, channels, frequency))

        split_feature = self.feature_split_layer(merge_feature)
        first_dim, channels, frequency = split_feature.shape
        split_feature = torch.reshape(split_feature, shape=(first_dim, channels, -1, 2))

        full_band_mask = self.full_band_decoder(split_feature[..., 0], full_band_encode_outs)

        sub_band_mask = self.sub_band_decoder(split_feature[..., 1], sub_band_encode_outs)

        full_band_mask = torch.reshape(full_band_mask, shape=(batch, frames, 2, -1))
        sub_band_mask = torch.reshape(sub_band_mask, shape=(batch, frames, 1, -1))
        return full_band_mask, sub_band_mask


    def forward(self, in_complex_spectrum: Tensor, in_amplitude_spectrum: Tensor, hidden_state: list):
        """
        :param in_amplitude_spectrum: (batch, frames, 1, frequency)
        :param hidden_state:
        :param in_complex_spectrum: (batch, frames, channels, frequency)
        :return:
        """
        merge_feature, full_band_encode_outs, sub_band_encode_outs = self.encode(in_complex_spectrum, in_amplitude_spectrum)

        out_hidden_state = list()

        for idx, rnn_layer in enumerate(self.dual_path_extension_rnn_list):
            merge_feature, state = rnn_layer(merge_feature, hidden_state[idx])
            out_hidden_state.append(state)

        full_band_mask, sub_band_mask = self.decode(merge_feature, full_band_encode_outs, sub_band_encode_outs)

        if self.need_mask:
            full_band_mask = self.mask_padding(full_band_mask)
        sub_band_mask = self.mask_padding(sub_band_mask)

        full_band_out = in_complex_spectrum * full_band_mask
        sub_band_out = in_amplitude_spectrum * sub_band_mask
        # outputs is (batch, frames, 2, frequency), complex style.

        if self.print_mask:
            self.print_spec(full_band_mask, sub_band_mask, full_band_out, sub_band_out)

        return full_band_out, sub_band_out, out_hidden_state

    
class FullSubPathExtension_ext(FullSubPathExtension):
    def __init__(self, configs: TrainConfig, need_mask: bool = True, print_mask: bool = False):
        super().__init__(configs, full_band_encoder=FullBandEncoder, 
                         sub_band_encoder=SubBandEncoder_ext, full_band_decoder=FullBandDecoder, 
                         sub_band_decoder=SubBandDecoder_ext, need_mask=need_mask, print_mask=print_mask)

    def forward(self, in_complex_spectrum: Tensor, in_amplitude_spectrum: Tensor, hidden_state: list):
        """
        :param in_amplitude_spectrum: (batch, frames, 1, frequency)
        :param hidden_state:
        :param in_complex_spectrum: (batch, frames, channels, frequency)
        :return:
        """
        merge_feature, full_band_encode_outs, sub_band_encode_outs = self.encode(in_complex_spectrum, in_amplitude_spectrum)

        out_hidden_state = list()

        for idx, rnn_layer in enumerate(self.dual_path_extension_rnn_list):
            merge_feature, state = rnn_layer(merge_feature, hidden_state[idx])
            out_hidden_state.append(state)

        full_band_mask, sub_band_mask = self.decode(merge_feature, full_band_encode_outs, sub_band_encode_outs)

        if self.need_mask:
            full_band_mask = self.mask_padding(full_band_mask)
        sub_band_mask = self.mask_padding(sub_band_mask)

        full_band_out = in_complex_spectrum * full_band_mask
        sub_band_out = in_amplitude_spectrum * sub_band_mask
        # outputs is (batch, frames, 2, frequency), complex style.

        full_band_out[:, :, 0:1, :] = (full_band_out[:, :, 0:1, :] + sub_band_out) / 2

        if self.print_mask:
            self.print_spec(full_band_mask, sub_band_mask, full_band_out, sub_band_out)

        return full_band_out, out_hidden_state
