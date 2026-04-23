import torch

from torch import nn, Tensor
import torch.autograd.profiler as profiler
# from models.en_decoder import FullBandEncoderBlock, FullBandDecoderBlock
from models.en_decoder import * #  SubBandEncoderBlock, SubBandDecoderBlock
from models.sequence_modules import DualPathExtensionRNN, DualPathExtensionRNNLight
from src.fspen_configs import TrainConfig, TrainConfigLarge, TrainConfigLarge1, TrainConfig_explicit

from functools import partial
from collections import OrderedDict

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


class FullBandEncoder(nn.Module):
    def __init__(self, configs: TrainConfig):
        super().__init__()

        last_channels = 0
        self.full_band_encoder = nn.ModuleList()
        for encoder_name, conv_parameter in configs.full_band_encoder.items():
            self.full_band_encoder.append(FullBandEncoderBlock(**conv_parameter))
            last_channels = conv_parameter["out_channels"]

        # global_feat_conv = nn.Conv1d # configs.full_band_encoder["encoder1"]["conv"]
        self.global_features = nn.Conv1d(in_channels=last_channels, out_channels=last_channels, kernel_size=1, stride=1)

    def forward(self, complex_spectrum: Tensor):
        """
        :param complex_spectrum: (batch*frame, channels, frequency)
        :return:
        """
        full_band_encodes = []
        for i, encoder in enumerate(self.full_band_encoder):
            # print(complex_spectrum.size())
            complex_spectrum = encoder(complex_spectrum)
            assert torch.isnan(complex_spectrum).any().item() is False, f"fullband enc conv_{i} out has NaNs"
            full_band_encodes.append(complex_spectrum)

        global_feature = self.global_features(complex_spectrum)
        assert torch.isnan(global_feature).any().item() is False, f"fullband global_feature out has NaNs"

        return full_band_encodes[::-1], global_feature


class SubBandEncoder(nn.Module):
    def __init__(self, configs: TrainConfig):
        super().__init__()

        self.sub_band_encoders = nn.ModuleList()
        for encoder_name, conv_parameters in configs.sub_band_encoder.items():
            self.sub_band_encoders.append(SubBandEncoderBlock(**conv_parameters["conv"]))

    def forward(self, amplitude_spectrum: Tensor):
        """
        :param amplitude_spectrum: (batch * frames, channels, frequency)
        :return:
        """
        sub_band_encodes = list()
        for encoder in self.sub_band_encoders:
            encode_out = encoder(amplitude_spectrum)
            sub_band_encodes.append(encode_out)
            # print(encode_out.shape)
        local_feature = torch.cat(sub_band_encodes, dim=2)  # feature cat

        return sub_band_encodes, local_feature
    

class SubBandEncoder_ver2(nn.Module):
    def __init__(self, configs: TrainConfig):
        super().__init__()

        self.sub_band_encoders = nn.ModuleList()
        self.freq_bounds = []
        for encoder_name, layer_parameters in configs.sub_band_encoder.items():
            self.freq_bounds.append(layer_parameters["bounds"])
            sub_band_layer = nn.ModuleList()
            for conv_parameters in layer_parameters["convs"]:
                sub_band_layer.append(FullBandEncoderBlock(**conv_parameters, normalize=False, is_sub=False))

            self.sub_band_encoders.append(sub_band_layer)

    def forward(self, amplitude_spectrum: Tensor):
        """
        :param amplitude_spectrum: (batch * frames, channels, frequency)
        :return:
        """
        sub_band_encodes = list()
        for ind, encoder in enumerate(self.sub_band_encoders):
            start_idx = self.freq_bounds[ind]["start_frequency"]
            end_idx = self.freq_bounds[ind]["end_frequency"]
            
            encode_in = amplitude_spectrum[:, :, start_idx: end_idx]
            conv_outs = []
            for i, conv in enumerate(encoder):
                encode_out = conv(encode_in)
                assert torch.isnan(encode_out).any().item() is False, f"subband enc conv {ind} group {i} layer out has NaNs"
                # print(start_idx, end_idx, encode_in.shape, encode_out.shape)
                conv_outs.append(encode_out)
                encode_in = encode_out

            sub_band_encodes.append(conv_outs)
            # print(encode_out.shape)
        local_feature = torch.cat([outs[-1] for outs in sub_band_encodes], dim=2)  # feature cat

        return sub_band_encodes, local_feature


class SubBandEncoder_ver3(nn.Module):
    def __init__(self, configs: TrainConfig):
        super().__init__()

        self.sub_band_encoders = nn.ModuleList()
        self.freq_bounds = []
        for encoder_name, layer_parameters in configs.sub_band_encoder.items():
            self.freq_bounds.append(layer_parameters["bounds"])
            sub_band_layer = nn.ModuleList()
            for conv_parameters in layer_parameters["convs"]:
                sub_band_layer.append(FullBandEncoderBlock(**conv_parameters, normalize=False, is_sub=False))

            self.sub_band_encoders.append(sub_band_layer)

        self.partition = configs.partition

    def forward(self, amplitude_spectrum: Tensor):
        """
        :param amplitude_spectrum: (batch * frames, chunks, channels, frequency)
        :return:
        """
        sub_band_encodes = list()
        first_dim, _, channels, _ = amplitude_spectrum.shape

        if self.partition is not None:
            bands = []
            start = 0
            for part in self.partition:
                x = amplitude_spectrum[:, start:start+part, :, :].permute(0, 2, 1, 3)
                x = x.reshape(first_dim, channels, -1)
                bands.append(x)

        for ind, encoder in enumerate(self.sub_band_encoders):
            # start_idx = self.freq_bounds[ind]["start_frequency"]
            # end_idx = self.freq_bounds[ind]["end_frequency"]
            encode_in = amplitude_spectrum[:, ind, :, :]
            if self.partition is not None:
                encode_in = bands[ind]
            conv_outs = []
            # print(encode_in.shape)
            for i, conv in enumerate(encoder):
                # print(encode_in.shape)
                encode_out = conv(encode_in)
                # print(encode_out.shape)
                assert torch.isnan(encode_out).any().item() is False, f"subband enc conv {ind} group {i} layer out has NaNs"
                # print(start_idx, end_idx, encode_in.shape, encode_out.shape)
                conv_outs.append(encode_out)
                encode_in = encode_out

            sub_band_encodes.append(conv_outs)
            # print(encode_out.shape)
        local_feature = torch.cat([outs[-1] for outs in sub_band_encodes], dim=2)  # feature cat

        return sub_band_encodes, local_feature
    

class SubBandEncoder_baseline(nn.Module):
    def __init__(self, configs: TrainConfig):
        super().__init__()

        self.sub_band_encoders = nn.ModuleList()
        for encoder_name, conv_parameters in configs.sub_band_encoder.items():
            self.sub_band_encoders.append(SubBandEncoderBlock_baseline(**conv_parameters["conv"]))

    def forward(self, amplitude_spectrum: Tensor):
        """
        :param amplitude_spectrum: (batch * frames, channels, frequency)
        :return:
        """
        sub_band_encodes = list()
        for encoder in self.sub_band_encoders:
            encode_out = encoder(amplitude_spectrum)
            sub_band_encodes.append(encode_out)
            # print(encode_out.shape)
        local_feature = torch.cat(sub_band_encodes, dim=2)  # feature cat

        return sub_band_encodes, local_feature


class FullBandDecoder(nn.Module):
    def __init__(self, configs: TrainConfig):
        super().__init__()
        self.full_band_decoders = nn.ModuleList()
        fbd_items = configs.full_band_decoder.items()
        for ind, (decoder_name, parameters) in enumerate(fbd_items):
            split_act = False
            if ind == len(fbd_items) - 1:
                split_act = True
            self.full_band_decoders.append(
                FullBandDecoderBlock(**parameters, split_act=split_act))

    def forward(self, feature: Tensor, encode_outs: list):
        for i, (decoder, encode_out) in enumerate(zip(self.full_band_decoders, encode_outs)):
            feature = decoder(feature, encode_out)
            assert torch.isnan(feature).any().item() is False, f"fullband dec conv_{i} out has NaNs"

        return feature


class FullBandDecoder_baseline(nn.Module):
    def __init__(self, configs: TrainConfig):
        super().__init__()
        self.full_band_decoders = nn.ModuleList()
        fbd_items = configs.full_band_decoder.items()
        for ind, (decoder_name, parameters) in enumerate(fbd_items):
            split_act = False
            if ind == len(fbd_items) - 1:
                split_act = True
            self.full_band_decoders.append(
                FullBandDecoderBlock(**parameters, split_act=False))

    def forward(self, feature: Tensor, encode_outs: list):
        for i, (decoder, encode_out) in enumerate(zip(self.full_band_decoders, encode_outs)):
            feature = decoder(feature, encode_out)
            assert torch.isnan(feature).any().item() is False, f"fullband dec conv_{i} out has NaNs"

        return feature


class FullBandDecoder_ver2(nn.Module):
    def __init__(self, configs: TrainConfig):
        super().__init__()
        self.full_band_decoders = nn.ModuleList()
        fbd_items = configs.full_band_decoder.items()
        for ind, (decoder_name, parameters) in enumerate(fbd_items):
            split_act = False
            if ind == len(fbd_items) - 1:
                split_act = True
            self.full_band_decoders.append(
                FullBandDecoderBlock(**parameters, split_act=split_act, mag_act=nn.Sigmoid))

    def forward(self, feature: Tensor, encode_outs: list):
        for i, (decoder, encode_out) in enumerate(zip(self.full_band_decoders, encode_outs)):
            feature = decoder(feature, encode_out)
            assert torch.isnan(feature).any().item() is False, f"fullband dec conv_{i} out has NaNs"

        return feature


class SubBandDecoder(nn.Module):
    def __init__(self, configs: TrainConfig):
        super().__init__()
        start_idx = 0
        self.sub_band_decoders = nn.ModuleList()
        for (decoder_name, parameters), bands in zip(configs.sub_band_decoder.items(), configs.bands_num_in_groups):
            end_idx = start_idx + bands
            self.sub_band_decoders.append(SubBandDecoderBlock(start_idx=start_idx, end_idx=end_idx, **parameters))
            start_idx = end_idx

    def forward(self, feature: Tensor, sub_encodes: list):
        """
        :param feature: (batch*frames, channels, bands)
        :param sub_encodes: [sub_encode_0, sub_encode_1, ...], each element is (batch*frames, channels, sub_bands)
        :return: (batch*frames, full-frequency)
        """
        sub_decoder_outs = []
        for decoder, sub_encode in zip(self.sub_band_decoders, sub_encodes):
            sub_decoder_out = decoder(feature, sub_encode)
            sub_decoder_outs.append(sub_decoder_out)

        sub_decoder_outs = torch.cat(tensors=sub_decoder_outs, dim=1)  # feature cat

        return sub_decoder_outs


class SubBandDecoder_ver2(nn.Module):
    def __init__(self, configs: TrainConfig):
        super().__init__()
        
        start_idx = 0
        self.bands = []
        self.sub_band_decoders = nn.ModuleList()
        sbd_items = configs.sub_band_decoder.items()
        output_dim = 0

        self.dims = []
        sbe_items = configs.sub_band_encoder
        self.overlap = configs.overlap
        for i, (decoder_name, layer_parameters) in enumerate(sbd_items):
            self.bands.append({"start": start_idx, "end": start_idx + layer_parameters["width"]})
            start_idx = start_idx + layer_parameters["width"]
            sub_band_layer = nn.ModuleList()
            x = torch.zeros(1, configs.sub_band_encoder["encoder1"]["convs"][-1]["out_channels"], layer_parameters["width"])
            for ind, conv_parameters in enumerate(layer_parameters["convs"]):
                is_sub = False
                if ind == len(layer_parameters["convs"]) - 1:
                    is_sub = True
                sub_band_layer.append(FullBandDecoderBlock(**conv_parameters, normalize=False, split_act=False, is_sub=is_sub))
                with torch.no_grad():
                    y = torch.zeros_like(x)
                    x = sub_band_layer[-1](x, y)

            if self.overlap:
                self.dims.append(sbe_items[f"encoder{i + 1}"]["old_group_width"])
            
            output_dim += x.shape[-1]
            self.sub_band_decoders.append(sub_band_layer)

        self.overlap = configs.overlap
        if self.overlap:
            self.matcher = nn.Sequential(
                nn.Linear(output_dim, configs.n_fft // 2),
                nn.ReLU(),
                nn.Linear(configs.n_fft // 2, configs.n_fft // 2)
            )
        
        # self.output_ch = configs.n_fft // 2
        # self.overlap = configs.overlap
        # if self.overlap is not None:
        #     self.output_conv = nn.Conv1d(1, 1, kernel_size=4, stride=2, padding=2)

    def forward(self, feature: Tensor, sub_encodes: list):
        """
        :param feature: (batch*frames, channels, bands)
        :param sub_encodes: [sub_encode_0, sub_encode_1, ...], each element is (batch*frames, channels, sub_bands)
        :return: (batch*frames, full-frequency)
        """
        sub_decoder_outs = []
        for idx, (decoder, sub_encode) in enumerate(zip(self.sub_band_decoders, sub_encodes)):
            start_idx = self.bands[idx]["start"]
            end_idx = self.bands[idx]["end"]
            decode_in = feature[:, :, start_idx: end_idx]
            for i, conv in enumerate(decoder):
                sub_decoder_out = conv(decode_in, sub_encode[len(decoder) - i - 1])
                assert torch.isnan(sub_decoder_out).any().item() is False, f"subband dec conv {idx} group {i} layer out has NaNs"
                decode_in = sub_decoder_out

            first_dim, bands, band_width = sub_decoder_out.shape
            sub_decoder_out = torch.reshape(sub_decoder_out, shape=(first_dim, bands*band_width))

            sub_decoder_outs.append(sub_decoder_out)

        sub_decoder_outs = torch.cat(tensors=sub_decoder_outs, dim=1)  # feature cat

        return sub_decoder_outs


class SubBandDecoder_ver3(nn.Module):
    def __init__(self, configs: TrainConfig):
        super().__init__()
        
        start_idx = 0
        self.bands = []
        self.sub_band_decoders = nn.ModuleList()
        sbd_items = configs.sub_band_decoder.items()
        output_dim = 0

        self.dims = []
        sbe_items = configs.sub_band_encoder
        self.overlap = configs.overlap
        for i, (decoder_name, layer_parameters) in enumerate(sbd_items):
            self.bands.append({"start": start_idx, "end": start_idx + layer_parameters["width"]})
            start_idx = start_idx + layer_parameters["width"]
            sub_band_layer = nn.ModuleList()
            x = torch.zeros(1, configs.sub_band_encoder["encoder1"]["convs"][-1]["out_channels"], layer_parameters["width"])
            for ind, conv_parameters in enumerate(layer_parameters["convs"]):
                is_sub = False
                if ind == len(layer_parameters["convs"]) - 1:
                    is_sub = True
                sub_band_layer.append(FullBandDecoderBlock(**conv_parameters, normalize=False, split_act=False, is_sub=is_sub, mag_act=nn.Sigmoid))
                with torch.no_grad():
                    y = torch.zeros_like(x)
                    x = sub_band_layer[-1](x, y)

            if self.overlap:
                self.dims.append(sbe_items[f"encoder{i + 1}"]["old_group_width"])
            
            output_dim += x.shape[-1]
            self.sub_band_decoders.append(sub_band_layer)

        self.overlap = configs.overlap
        if self.overlap:
            self.matcher = nn.Sequential(
                nn.Linear(output_dim, configs.n_fft // 2),
                nn.ReLU(),
                nn.Linear(configs.n_fft // 2, configs.n_fft // 2)
            )
        
        # self.output_ch = configs.n_fft // 2
        # self.overlap = configs.overlap
        # if self.overlap is not None:
        #     self.output_conv = nn.Conv1d(1, 1, kernel_size=4, stride=2, padding=2)

    def forward(self, feature: Tensor, sub_encodes: list):
        """
        :param feature: (batch*frames, channels, bands)
        :param sub_encodes: [sub_encode_0, sub_encode_1, ...], each element is (batch*frames, channels, sub_bands)
        :return: (batch*frames, full-frequency)
        """
        sub_decoder_outs = []
        # print(feature.shape)
        for idx, (decoder, sub_encode) in enumerate(zip(self.sub_band_decoders, sub_encodes)):
            start_idx = self.bands[idx]["start"]
            end_idx = self.bands[idx]["end"]
            # print(start_idx, end_idx)
            decode_in = feature[:, :, start_idx:end_idx]
            # print(feature.shape)
            for i, conv in enumerate(decoder):
                # print(decode_in.shape, sub_encode[len(decoder) - i - 1].shape)
                # print(decode_in.shape, sub_encode[len(decoder) - i - 1].shape)
                sub_decoder_out = conv(decode_in, sub_encode[len(decoder) - i - 1])

                assert torch.isnan(sub_decoder_out).any().item() is False, f"subband dec conv {idx} group {i} layer out has NaNs"
                decode_in = sub_decoder_out

            first_dim, bands, band_width = sub_decoder_out.shape
            sub_decoder_out = torch.reshape(sub_decoder_out, shape=(first_dim, bands*band_width))
            # print(sub_decoder_out.shape)
            # if self.overlap:
            #     sub_decoder_out = sub_decoder_out[..., :self.dims[idx]]
            sub_decoder_outs.append(sub_decoder_out)

        # print(sub_decoder_outs[0].shape)
        sub_decoder_outs = torch.cat(tensors=sub_decoder_outs, dim=1)  # feature cat

        return sub_decoder_outs


class FullSubPathExtension(nn.Module):
    def __init__(self, configs: TrainConfig, need_mask: bool = True):
        super().__init__()
        self.full_band_encoder = FullBandEncoder(configs)
        self.sub_band_encoder = SubBandEncoder_baseline(configs) # SubBandEncoder_ver2(configs)
        # self.sub_band_encoder = SubBandEncoder_ver2(configs)
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

        # with profiler.record_function("Create GRU"):
        self.dual_path_extension_rnn_list = nn.ModuleList()
        for _ in range(configs.dual_path_extension["num_modules"]):
            self.dual_path_extension_rnn_list.append(DualPathExtensionRNN(**configs.dual_path_extension["parameters"]))

        self.feature_split_layer = nn.Sequential(
            nn.Conv1d(in_channels=merge_bands//compress_rate, out_channels=merge_bands, kernel_size=1, stride=1),
            nn.Linear(in_features=merge_channels//compress_rate, out_features=merge_channels),
            nn.ELU()
        )

        self.full_band_decoder = FullBandDecoder_baseline(configs)
        self.sub_band_decoder = SubBandDecoder(configs) # SubBandDecoder_ver2(configs)
        # self.sub_band_decoder = SubBandDecoder_ver2(configs)

        self.mask_padding = nn.ConstantPad2d(padding=(1, 0, 0, 0), value=0.0)
        self.need_mask = need_mask

    def forward(self, in_complex_spectrum: Tensor, in_amplitude_spectrum: Tensor, hidden_state: list):
        """
        :param in_amplitude_spectrum: (batch, frames, 1, frequency)
        :param hidden_state:
        :param in_complex_spectrum: (batch, frames, channels, frequency)
        :return:
        """
        batch, frames, channels, frequency = in_complex_spectrum.shape
        # 16 // 8 for trainconfig
        # with profiler.record_function("Hidden state gen"):
        # hidden_state = [[torch.randn(1, batch * 32, 16 // 8, device=in_complex_spectrum.device) for _ in range(8)] for _ in range(self.num_rnn_modules)] # for rnn2 batch * 32 // 2
        complex_spectrum = torch.reshape(in_complex_spectrum, shape=(batch * frames, channels, frequency))
        amplitude_spectrum = torch.reshape(in_amplitude_spectrum, shape=(batch*frames, 1, frequency))
        # print("Complex Spectrum", complex_spectrum.shape)
        # with profiler.record_function("Full band encoder"):
        full_band_encode_outs, global_feature = self.full_band_encoder(complex_spectrum)
        # with profiler.record_function("Sub band encoder"):
        sub_band_encode_outs, local_feature = self.sub_band_encoder(amplitude_spectrum)
        # print(f"FBE out:", full_band_encode_outs.shape, "SBE out:", sub_band_encode_outs.shape)
        # print(global_feature.shape, local_feature.shape)
        merge_feature = torch.cat(tensors=[global_feature, local_feature], dim=2)  # feature cat
        # print(f"Merge layer in:", merge_feature.shape)
        merge_feature = self.feature_merge_layer(merge_feature)
        # print(f"Merge layer out:", merge_feature.shape)
        # (batch*frames, channels, frequency) -> (batch*frames, channels//2, frequency//2)
        _, channels, frequency = merge_feature.shape
        merge_feature = torch.reshape(merge_feature, shape=(batch, frames, channels, frequency))
        merge_feature = torch.permute(merge_feature, dims=(0, 3, 1, 2)).contiguous()
        # (batch, frequency, frames, channels)
        # with profiler.record_function("RNN layer"):
        out_hidden_state = list()
        # print(f"RNN in:", merge_feature.shape)
        for idx, rnn_layer in enumerate(self.dual_path_extension_rnn_list):
            merge_feature, state = rnn_layer(merge_feature, hidden_state[idx])
            out_hidden_state.append(state)
        # print(f"RNN out:", merge_feature.shape)
        merge_feature = torch.permute(merge_feature, dims=(0, 2, 3, 1)).contiguous()
        merge_feature = torch.reshape(merge_feature, shape=(batch * frames, channels, frequency))
        # print(f"Split layer in:", merge_feature.shape)
        split_feature = self.feature_split_layer(merge_feature)
        first_dim, channels, frequency = split_feature.shape
        split_feature = torch.reshape(split_feature, shape=(first_dim, channels, -1, 2))
        # print(f"Split layer out:", split_feature.shape)
        # print(f"FBD in:", split_feature[..., 0].shape,)# full_band_encode_outs.shape)
        # print(f"SBD in:", split_feature[..., 1].shape,)# sub_band_encode_outs.shape)
        # with profiler.record_function("Full band decoder"):
        # print(split_feature[..., 0].shape)
        full_band_mask = self.full_band_decoder(split_feature[..., 0], full_band_encode_outs)
        # with profiler.record_function("Sub band decoder"):
        sub_band_mask = self.sub_band_decoder(split_feature[..., 1], sub_band_encode_outs)

        full_band_mask = torch.reshape(full_band_mask, shape=(batch, frames, 2, -1))
        sub_band_mask = torch.reshape(sub_band_mask, shape=(batch, frames, 1, -1))

        # Zero padding in the DC signal part removes the DC component
        # with profiler.record_function("Mask padding"):
        if self.need_mask:
            full_band_mask = self.mask_padding(full_band_mask) # uncomment for all modeles except TrainConfig48kHzEnc2x_ver1
        sub_band_mask = self.mask_padding(sub_band_mask)
        # print(in_complex_spectrum.shape, full_band_mask.shape)
        full_band_out = in_complex_spectrum * full_band_mask
        sub_band_out = in_amplitude_spectrum * sub_band_mask
        # outputs is (batch, frames, 2, frequency), complex style.

        # full_band_out[:, :, 0:1, :] = (full_band_out[:, :, 0:1, :] + sub_band_out) / 2
        return full_band_out, sub_band_out, out_hidden_state

class FullSubPathExtension_abs_pha(nn.Module):
    def __init__(self, configs: TrainConfig, need_mask: bool = True):
        super().__init__()
        self.full_band_encoder = FullBandEncoder(configs)
        self.sub_band_encoder = SubBandEncoder(configs) # SubBandEncoder_ver2(configs)
        # self.sub_band_encoder = SubBandEncoder_ver2(configs)
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

        # with profiler.record_function("Create GRU"):
        self.dual_path_extension_rnn_list = nn.ModuleList()
        for _ in range(configs.dual_path_extension["num_modules"]):
            self.dual_path_extension_rnn_list.append(DualPathExtensionRNN(**configs.dual_path_extension["parameters"]))

        self.feature_split_layer = nn.Sequential(
            nn.Conv1d(in_channels=merge_bands//compress_rate, out_channels=merge_bands, kernel_size=1, stride=1),
            nn.Linear(in_features=merge_channels//compress_rate, out_features=merge_channels),
            nn.ELU()
        )

        self.full_band_decoder = FullBandDecoder(configs)
        self.sub_band_decoder = SubBandDecoder(configs) # SubBandDecoder_ver2(configs)
        # self.sub_band_decoder = SubBandDecoder_ver2(configs)

        self.mask_padding = nn.ConstantPad2d(padding=(1, 0, 0, 0), value=0.0)
        self.need_mask = need_mask

    def forward(self, in_complex_spectrum: Tensor, in_amplitude_spectrum: Tensor, hidden_state: list):
        """
        :param in_amplitude_spectrum: (batch, frames, 1, frequency)
        :param hidden_state:
        :param in_complex_spectrum: (batch, frames, channels, frequency)
        :return:
        """
        batch, frames, channels, frequency = in_complex_spectrum.shape
        # 16 // 8 for trainconfig
        # with profiler.record_function("Hidden state gen"):
        # hidden_state = [[torch.randn(1, batch * 32, 16 // 8, device=in_complex_spectrum.device) for _ in range(8)] for _ in range(self.num_rnn_modules)] # for rnn2 batch * 32 // 2
        complex_spectrum = torch.reshape(in_complex_spectrum, shape=(batch * frames, channels, frequency))
        amplitude_spectrum = torch.reshape(in_amplitude_spectrum, shape=(batch*frames, 1, frequency))
        # print("Complex Spectrum", complex_spectrum.shape)
        # with profiler.record_function("Full band encoder"):
        full_band_encode_outs, global_feature = self.full_band_encoder(complex_spectrum)
        # with profiler.record_function("Sub band encoder"):
        sub_band_encode_outs, local_feature = self.sub_band_encoder(amplitude_spectrum)
        # print(f"FBE out:", full_band_encode_outs.shape, "SBE out:", sub_band_encode_outs.shape)
        # print(global_feature.shape, local_feature.shape)
        merge_feature = torch.cat(tensors=[global_feature, local_feature], dim=2)  # feature cat
        # print(f"Merge layer in:", merge_feature.shape)
        merge_feature = self.feature_merge_layer(merge_feature)
        # print(f"Merge layer out:", merge_feature.shape)
        # (batch*frames, channels, frequency) -> (batch*frames, channels//2, frequency//2)
        _, channels, frequency = merge_feature.shape
        merge_feature = torch.reshape(merge_feature, shape=(batch, frames, channels, frequency))
        merge_feature = torch.permute(merge_feature, dims=(0, 3, 1, 2)).contiguous()
        # (batch, frequency, frames, channels)
        # with profiler.record_function("RNN layer"):
        out_hidden_state = list()
        # print(f"RNN in:", merge_feature.shape)
        for idx, rnn_layer in enumerate(self.dual_path_extension_rnn_list):
            merge_feature, state = rnn_layer(merge_feature, hidden_state[idx])
            out_hidden_state.append(state)
        # print(f"RNN out:", merge_feature.shape)
        merge_feature = torch.permute(merge_feature, dims=(0, 2, 3, 1)).contiguous()
        merge_feature = torch.reshape(merge_feature, shape=(batch * frames, channels, frequency))
        # print(f"Split layer in:", merge_feature.shape)
        split_feature = self.feature_split_layer(merge_feature)
        first_dim, channels, frequency = split_feature.shape
        split_feature = torch.reshape(split_feature, shape=(first_dim, channels, -1, 2))

        full_band_mask = self.full_band_decoder(split_feature[..., 0], full_band_encode_outs)
        # with profiler.record_function("Sub band decoder"):
        sub_band_mask = self.sub_band_decoder(split_feature[..., 1], sub_band_encode_outs)

        full_band_mask = torch.reshape(full_band_mask, shape=(batch, frames, 2, -1))
        sub_band_mask = torch.reshape(sub_band_mask, shape=(batch, frames, 1, -1))

        # Zero padding in the DC signal part removes the DC component
        # with profiler.record_function("Mask padding"):
        if self.need_mask:
            full_band_mask = self.mask_padding(full_band_mask) # uncomment for all modeles except TrainConfig48kHzEnc2x_ver1
        sub_band_mask = self.mask_padding(sub_band_mask)
        # print(in_complex_spectrum.shape, full_band_mask.shape)
        full_band_out = in_complex_spectrum * full_band_mask
        sub_band_out = in_amplitude_spectrum * sub_band_mask
        # outputs is (batch, frames, 2, frequency), complex style.

        full_band_out[:, :, 0:1, :] = (full_band_out[:, :, 0:1, :] + sub_band_out) / 2
        return full_band_out, out_hidden_state
    
class FullSubPathExtension_abs_pha_mapping(nn.Module):
    def __init__(self, configs: TrainConfig, need_mask: bool = True):
        super().__init__()
        self.full_band_encoder = FullBandEncoder(configs)
        self.sub_band_encoder = SubBandEncoder(configs) # SubBandEncoder_ver2(configs)
        # self.sub_band_encoder = SubBandEncoder_ver2(configs)
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

        # with profiler.record_function("Create GRU"):
        self.dual_path_extension_rnn_list = nn.ModuleList()
        for _ in range(configs.dual_path_extension["num_modules"]):
            self.dual_path_extension_rnn_list.append(DualPathExtensionRNN(**configs.dual_path_extension["parameters"]))

        self.feature_split_layer = nn.Sequential(
            nn.Conv1d(in_channels=merge_bands//compress_rate, out_channels=merge_bands, kernel_size=1, stride=1),
            nn.Linear(in_features=merge_channels//compress_rate, out_features=merge_channels),
            nn.ELU()
        )

        self.full_band_decoder = FullBandDecoder(configs)
        self.sub_band_decoder = SubBandDecoder(configs) # SubBandDecoder_ver2(configs)
        # self.sub_band_decoder = SubBandDecoder_ver2(configs)

        self.mask_padding = nn.ConstantPad2d(padding=(1, 0, 0, 0), value=0.0)
        self.need_mask = need_mask

    def forward(self, in_complex_spectrum: Tensor, in_amplitude_spectrum: Tensor, hidden_state: list):
        """
        :param in_amplitude_spectrum: (batch, frames, 1, frequency)
        :param hidden_state:
        :param in_complex_spectrum: (batch, frames, channels, frequency)
        :return:
        """
        batch, frames, channels, frequency = in_complex_spectrum.shape
        # 16 // 8 for trainconfig
        # with profiler.record_function("Hidden state gen"):
        # hidden_state = [[torch.randn(1, batch * 32, 16 // 8, device=in_complex_spectrum.device) for _ in range(8)] for _ in range(self.num_rnn_modules)] # for rnn2 batch * 32 // 2
        complex_spectrum = torch.reshape(in_complex_spectrum, shape=(batch * frames, channels, frequency))
        amplitude_spectrum = torch.reshape(in_amplitude_spectrum, shape=(batch*frames, 1, frequency))
        # print("Complex Spectrum", complex_spectrum.shape)
        # with profiler.record_function("Full band encoder"):
        full_band_encode_outs, global_feature = self.full_band_encoder(complex_spectrum)
        # with profiler.record_function("Sub band encoder"):
        sub_band_encode_outs, local_feature = self.sub_band_encoder(amplitude_spectrum)
        # print(f"FBE out:", full_band_encode_outs.shape, "SBE out:", sub_band_encode_outs.shape)
        # print(global_feature.shape, local_feature.shape)
        merge_feature = torch.cat(tensors=[global_feature, local_feature], dim=2)  # feature cat
        # print(f"Merge layer in:", merge_feature.shape)
        merge_feature = self.feature_merge_layer(merge_feature)
        # print(f"Merge layer out:", merge_feature.shape)
        # (batch*frames, channels, frequency) -> (batch*frames, channels//2, frequency//2)
        _, channels, frequency = merge_feature.shape
        merge_feature = torch.reshape(merge_feature, shape=(batch, frames, channels, frequency))
        merge_feature = torch.permute(merge_feature, dims=(0, 3, 1, 2)).contiguous()
        # (batch, frequency, frames, channels)
        # with profiler.record_function("RNN layer"):
        out_hidden_state = list()
        # print(f"RNN in:", merge_feature.shape)
        for idx, rnn_layer in enumerate(self.dual_path_extension_rnn_list):
            merge_feature, state = rnn_layer(merge_feature, hidden_state[idx])
            out_hidden_state.append(state)
        # print(f"RNN out:", merge_feature.shape)
        merge_feature = torch.permute(merge_feature, dims=(0, 2, 3, 1)).contiguous()
        merge_feature = torch.reshape(merge_feature, shape=(batch * frames, channels, frequency))
        # print(f"Split layer in:", merge_feature.shape)
        split_feature = self.feature_split_layer(merge_feature)
        first_dim, channels, frequency = split_feature.shape
        split_feature = torch.reshape(split_feature, shape=(first_dim, channels, -1, 2))

        full_band_mask = self.full_band_decoder(split_feature[..., 0], full_band_encode_outs)
        # with profiler.record_function("Sub band decoder"):
        sub_band_mask = self.sub_band_decoder(split_feature[..., 1], sub_band_encode_outs)

        full_band_mask = torch.reshape(full_band_mask, shape=(batch, frames, 2, -1))
        sub_band_mask = torch.reshape(sub_band_mask, shape=(batch, frames, 1, -1))

        # Zero padding in the DC signal part removes the DC component
        # with profiler.record_function("Mask padding"):
        if self.need_mask:
            full_band_mask = self.mask_padding(full_band_mask) # uncomment for all modeles except TrainConfig48kHzEnc2x_ver1
        sub_band_mask = self.mask_padding(sub_band_mask)
        # print(in_complex_spectrum.shape, full_band_mask.shape)
        full_band_out = in_complex_spectrum * full_band_mask
        sub_band_out = sub_band_mask # in_amplitude_spectrum * sub_band_mask
        # outputs is (batch, frames, 2, frequency), complex style.

        full_band_out[:, :, 0:1, :] = (full_band_out[:, :, 0:1, :] + sub_band_out) / 2
        return full_band_out, out_hidden_state

class FullSubPathExtension_ver2(nn.Module):
    def __init__(self, configs: TrainConfig, need_mask: bool = True):
        super().__init__()
        self.full_band_encoder = FullBandEncoder(configs)
        # self.sub_band_encoder = SubBandEncoder(configs) # SubBandEncoder_ver2(configs)
        self.sub_band_encoder = SubBandEncoder_ver2(configs)
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

        # with profiler.record_function("Create GRU"):
        self.dual_path_extension_rnn_list = nn.ModuleList()
        for _ in range(configs.dual_path_extension["num_modules"]):
            self.dual_path_extension_rnn_list.append(DualPathExtensionRNN(**configs.dual_path_extension["parameters"]))

        self.feature_split_layer = nn.Sequential(
            nn.Conv1d(in_channels=merge_bands//compress_rate, out_channels=merge_bands, kernel_size=1, stride=1),
            nn.Linear(in_features=merge_channels//compress_rate, out_features=merge_channels),
            nn.ELU()
        )

        self.full_band_decoder = FullBandDecoder(configs)
        # self.sub_band_decoder = SubBandDecoder(configs) # SubBandDecoder_ver2(configs)
        self.sub_band_decoder = SubBandDecoder_ver2(configs)

        self.mask_padding = nn.ConstantPad2d(padding=(1, 0, 0, 0), value=0.0)
        self.need_mask = need_mask

    def forward(self, in_complex_spectrum: Tensor, in_amplitude_spectrum: Tensor, hidden_state: list):
        """
        :param in_amplitude_spectrum: (batch, frames, 1, frequency)
        :param hidden_state:
        :param in_complex_spectrum: (batch, frames, channels, frequency)
        :return:
        """
        batch, frames, channels, frequency = in_complex_spectrum.shape
        # 16 // 8 for trainconfig
        # with profiler.record_function("Hidden state gen"):
        # hidden_state = [[torch.randn(1, batch * 32, 16 // 8, device=in_complex_spectrum.device) for _ in range(8)] for _ in range(self.num_rnn_modules)] # for rnn2 batch * 32 // 2
        complex_spectrum = torch.reshape(in_complex_spectrum, shape=(batch * frames, channels, frequency))
        amplitude_spectrum = torch.reshape(in_amplitude_spectrum, shape=(batch*frames, 1, frequency))
        # print("Complex Spectrum", complex_spectrum.shape)
        # with profiler.record_function("Full band encoder"):
        full_band_encode_outs, global_feature = self.full_band_encoder(complex_spectrum)
        # with profiler.record_function("Sub band encoder"):
        sub_band_encode_outs, local_feature = self.sub_band_encoder(amplitude_spectrum)
        # print(f"FBE out:", full_band_encode_outs.shape, "SBE out:", sub_band_encode_outs.shape)
        # print(global_feature.shape, local_feature.shape)
        merge_feature = torch.cat(tensors=[global_feature, local_feature], dim=2)  # feature cat
        # print(f"Merge layer in:", merge_feature.shape)
        merge_feature = self.feature_merge_layer(merge_feature)
        # print(f"Merge layer out:", merge_feature.shape)
        # (batch*frames, channels, frequency) -> (batch*frames, channels//2, frequency//2)
        _, channels, frequency = merge_feature.shape
        merge_feature = torch.reshape(merge_feature, shape=(batch, frames, channels, frequency))
        merge_feature = torch.permute(merge_feature, dims=(0, 3, 1, 2)).contiguous() # (batch, frequency, frames, channels)
        # (batch, frequency, frames, channels)
        # with profiler.record_function("RNN layer"):
        out_hidden_state = list()
        # print(f"RNN in:", merge_feature.shape)
        for idx, rnn_layer in enumerate(self.dual_path_extension_rnn_list):
            merge_feature, state = rnn_layer(merge_feature, hidden_state[idx])
            out_hidden_state.append(state)
        # print(f"RNN out:", merge_feature.shape)
        merge_feature = torch.permute(merge_feature, dims=(0, 2, 3, 1)).contiguous()
        merge_feature = torch.reshape(merge_feature, shape=(batch * frames, channels, frequency))
        # print(f"Split layer in:", merge_feature.shape)
        split_feature = self.feature_split_layer(merge_feature)
        first_dim, channels, frequency = split_feature.shape
        split_feature = torch.reshape(split_feature, shape=(first_dim, channels, -1, 2))
        # print(f"Split layer out:", split_feature.shape)
        # print(f"FBD in:", split_feature[..., 0].shape,)# full_band_encode_outs.shape)
        # print(f"SBD in:", split_feature[..., 1].shape,)# sub_band_encode_outs.shape)
        # with profiler.record_function("Full band decoder"):
        # print(split_feature[..., 0].shape)
        full_band_mask = self.full_band_decoder(split_feature[..., 0], full_band_encode_outs)
        # with profiler.record_function("Sub band decoder"):
        sub_band_mask = self.sub_band_decoder(split_feature[..., 1], sub_band_encode_outs)

        full_band_mask = torch.reshape(full_band_mask, shape=(batch, frames, 2, -1))
        sub_band_mask = torch.reshape(sub_band_mask, shape=(batch, frames, 1, -1))

        # Zero padding in the DC signal part removes the DC component
        # with profiler.record_function("Mask padding"):
        if self.need_mask:
            full_band_mask = self.mask_padding(full_band_mask) # uncomment for all modeles except TrainConfig48kHzEnc2x_ver1
        sub_band_mask = self.mask_padding(sub_band_mask)
        # print(in_complex_spectrum.shape, full_band_mask.shape)
        full_band_out = in_complex_spectrum * full_band_mask
        sub_band_out = in_amplitude_spectrum * sub_band_mask
        # outputs is (batch, frames, 2, frequency), complex style.

        # full_band_out[:, :, 0:1, :] = (full_band_out[:, :, 0:1, :] + sub_band_out) / 2
        return full_band_out, sub_band_out, out_hidden_state
    
class FullSubPathExtension_ver2_abs_pha(nn.Module):
    def __init__(self, configs: TrainConfig, need_mask: bool = True, print_mask: bool = False):
        super().__init__()
        self.full_band_encoder = FullBandEncoder(configs)
        # self.sub_band_encoder = SubBandEncoder(configs) # SubBandEncoder_ver2(configs)
        self.sub_band_encoder = SubBandEncoder_ver2(configs)
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

        # with profiler.record_function("Create GRU"):
        self.dual_path_extension_rnn_list = nn.ModuleList()
        for _ in range(configs.dual_path_extension["num_modules"]):
            self.dual_path_extension_rnn_list.append(DualPathExtensionRNN(**configs.dual_path_extension["parameters"]))

        self.feature_split_layer = nn.Sequential(
            nn.Conv1d(in_channels=merge_bands//compress_rate, out_channels=merge_bands, kernel_size=1, stride=1),
            nn.Linear(in_features=merge_channels//compress_rate, out_features=merge_channels),
            nn.ELU()
        )

        self.full_band_decoder = FullBandDecoder(configs)
        # self.sub_band_decoder = SubBandDecoder(configs) # SubBandDecoder_ver2(configs)
        self.sub_band_decoder = SubBandDecoder_ver2(configs)

        self.mask_padding = nn.ConstantPad2d(padding=(1, 0, 0, 0), value=0.0)
        self.need_mask = need_mask
        self.print_mask = print_mask

    @staticmethod
    def print_spec(fb_mask, sb_mask, fb_out, sb_out):
        plt.figure(figsize=(20, 10))
        plt.subplot(2, 2, 1)
        plt.imshow(fb_mask[0, :, 0].detach().permute(1, 0).numpy(), aspect='auto', origin='lower', norm=LogNorm(vmin=1e-3, vmax=1))
        plt.title("Full Band Mask")
        plt.colorbar()
        plt.subplot(2, 2, 2)
        plt.imshow(sb_mask[0, :, 0].detach().permute(1, 0).numpy(), aspect='auto', origin='lower', norm=LogNorm(vmin=1e-3, vmax=1))
        plt.title("Sub Band Mask")
        plt.colorbar()
        plt.subplot(2, 2, 3)
        plt.imshow(fb_out[0, :, 0].detach().permute(1, 0).numpy(), aspect='auto', origin='lower', norm=LogNorm(vmin=1e-3, vmax=1))
        plt.title("Full Band Masked Output")
        plt.colorbar()
        plt.subplot(2, 2, 4)
        plt.imshow(sb_out[0, :, 0].detach().permute(1, 0).numpy(), aspect='auto', origin='lower', norm=LogNorm(vmin=1e-3, vmax=1))
        plt.title("Sub Band Masked Output")
        plt.colorbar()

    def forward(self, in_complex_spectrum: Tensor, in_amplitude_spectrum: Tensor, hidden_state: list):
        """
        :param in_amplitude_spectrum: (batch, frames, 1, frequency)
        :param hidden_state:
        :param in_complex_spectrum: (batch, frames, channels, frequency)
        :return:
        """
        batch, frames, channels, frequency = in_complex_spectrum.shape
        # 16 // 8 for trainconfig
        # with profiler.record_function("Hidden state gen"):
        # hidden_state = [[torch.randn(1, batch * 32, 16 // 8, device=in_complex_spectrum.device) for _ in range(8)] for _ in range(self.num_rnn_modules)] # for rnn2 batch * 32 // 2
        complex_spectrum = torch.reshape(in_complex_spectrum, shape=(batch * frames, channels, frequency))
        amplitude_spectrum = torch.reshape(in_amplitude_spectrum, shape=(batch * frames, 1, frequency))
        # print("Complex Spectrum", complex_spectrum.shape)
        # with profiler.record_function("Full band encoder"):
        full_band_encode_outs, global_feature = self.full_band_encoder(complex_spectrum)
        # with profiler.record_function("Sub band encoder"):
        sub_band_encode_outs, local_feature = self.sub_band_encoder(amplitude_spectrum)
        # print(f"FBE out:", full_band_encode_outs.shape, "SBE out:", sub_band_encode_outs.shape)
        # print(global_feature.shape, local_feature.shape)
        merge_feature = torch.cat(tensors=[global_feature, local_feature], dim=2)  # feature cat
        # print(f"Merge layer in:", merge_feature.shape)
        merge_feature = self.feature_merge_layer(merge_feature)
        # print(f"Merge layer out:", merge_feature.shape)
        # (batch*frames, channels, frequency) -> (batch*frames, channels//2, frequency//2)
        _, channels, frequency = merge_feature.shape
        merge_feature = torch.reshape(merge_feature, shape=(batch, frames, channels, frequency))
        merge_feature = torch.permute(merge_feature, dims=(0, 3, 1, 2)).contiguous()
        # (batch, frequency, frames, channels)
        # with profiler.record_function("RNN layer"):
        out_hidden_state = list()
        # print(f"RNN in:", merge_feature.shape)
        for idx, rnn_layer in enumerate(self.dual_path_extension_rnn_list):
            merge_feature, state = rnn_layer(merge_feature, hidden_state[idx])
            out_hidden_state.append(state)
        # print(f"RNN out:", merge_feature.shape)
        merge_feature = torch.permute(merge_feature, dims=(0, 2, 3, 1)).contiguous()
        merge_feature = torch.reshape(merge_feature, shape=(batch * frames, channels, frequency))
        # print(f"Split layer in:", merge_feature.shape)
        split_feature = self.feature_split_layer(merge_feature)
        first_dim, channels, frequency = split_feature.shape
        split_feature = torch.reshape(split_feature, shape=(first_dim, channels, -1, 2))
        # print(f"Split layer out:", split_feature.shape)
        # print(f"FBD in:", split_feature[..., 0].shape,)# full_band_encode_outs.shape)
        # print(f"SBD in:", split_feature[..., 1].shape,)# sub_band_encode_outs.shape)
        # with profiler.record_function("Full band decoder"):
        # print(split_feature[..., 0].shape)
        full_band_mask = self.full_band_decoder(split_feature[..., 0], full_band_encode_outs)
        # with profiler.record_function("Sub band decoder"):
        sub_band_mask = self.sub_band_decoder(split_feature[..., 1], sub_band_encode_outs)

        full_band_mask = torch.reshape(full_band_mask, shape=(batch, frames, 2, -1))
        sub_band_mask = torch.reshape(sub_band_mask, shape=(batch, frames, 1, -1))

        # Zero padding in the DC signal part removes the DC component
        # with profiler.record_function("Mask padding"):
        if self.need_mask:
            full_band_mask = self.mask_padding(full_band_mask) # uncomment for all modeles except TrainConfig48kHzEnc2x_ver1
        sub_band_mask = self.mask_padding(sub_band_mask)
        # print(in_complex_spectrum.shape, full_band_mask.shape)
        full_band_out = in_complex_spectrum * full_band_mask
        sub_band_out = in_amplitude_spectrum * sub_band_mask
        # outputs is (batch, frames, 2, frequency), complex style.

        full_band_out[:, :, 0:1, :] = (full_band_out[:, :, 0:1, :] + sub_band_out) / 2

        if self.print_mask:
            self.print_spec(full_band_mask, sub_band_mask, full_band_out, sub_band_out)

        return full_band_out, out_hidden_state
    

class FullSubPathExtension_ver2_abs_pha_no_merge(nn.Module):
    def __init__(self, configs: TrainConfig, need_mask: bool = True, print_mask: bool = False):
        super().__init__()
        self.full_band_encoder = FullBandEncoder(configs)
        # self.sub_band_encoder = SubBandEncoder(configs) # SubBandEncoder_ver2(configs)
        self.sub_band_encoder = SubBandEncoder_ver2(configs)
        self.num_rnn_modules = configs.dual_path_extension["num_modules"]

        # merge_split = configs.merge_split
        # merge_channels = merge_split["channels"]
        # merge_bands = merge_split["bands"]
        # compress_rate = merge_split["compress_rate"]

        # with profiler.record_function("Create GRU"):
        self.dual_path_extension_rnn_list = nn.ModuleList()
        for _ in range(configs.dual_path_extension["num_modules"]):
            self.dual_path_extension_rnn_list.append(DualPathExtensionRNN(**configs.dual_path_extension["parameters"]))

        self.full_band_decoder = FullBandDecoder(configs)
        # self.sub_band_decoder = SubBandDecoder(configs) # SubBandDecoder_ver2(configs)
        self.sub_band_decoder = SubBandDecoder_ver2(configs)

        self.mask_padding = nn.ConstantPad2d(padding=(1, 0, 0, 0), value=0.0)
        self.need_mask = need_mask
        self.print_mask = print_mask

    @staticmethod
    def print_spec(fb_mask, sb_mask, fb_out, sb_out):
        plt.figure(figsize=(20, 10))
        plt.subplot(2, 2, 1)
        plt.imshow(fb_mask[0, :, 0].detach().permute(1, 0).numpy(), aspect='auto', origin='lower', norm=LogNorm(vmin=1e-3, vmax=1))
        plt.title("Full Band Mask")
        plt.colorbar()
        plt.subplot(2, 2, 2)
        plt.imshow(sb_mask[0, :, 0].detach().permute(1, 0).numpy(), aspect='auto', origin='lower', norm=LogNorm(vmin=1e-3, vmax=1))
        plt.title("Sub Band Mask")
        plt.colorbar()
        plt.subplot(2, 2, 3)
        plt.imshow(fb_out[0, :, 0].detach().permute(1, 0).numpy(), aspect='auto', origin='lower', norm=LogNorm(vmin=1e-3, vmax=1))
        plt.title("Full Band Masked Output")
        plt.colorbar()
        plt.subplot(2, 2, 4)
        plt.imshow(sb_out[0, :, 0].detach().permute(1, 0).numpy(), aspect='auto', origin='lower', norm=LogNorm(vmin=1e-3, vmax=1))
        plt.title("Sub Band Masked Output")
        plt.colorbar()

    def forward(self, in_complex_spectrum: Tensor, in_amplitude_spectrum: Tensor, hidden_state: list):
        """
        :param in_amplitude_spectrum: (batch, frames, 1, frequency)
        :param hidden_state:
        :param in_complex_spectrum: (batch, frames, channels, frequency)
        :return:
        """
        batch, frames, channels, frequency = in_complex_spectrum.shape
        # 16 // 8 for trainconfig
        # with profiler.record_function("Hidden state gen"):
        # hidden_state = [[torch.randn(1, batch * 32, 16 // 8, device=in_complex_spectrum.device) for _ in range(8)] for _ in range(self.num_rnn_modules)] # for rnn2 batch * 32 // 2
        complex_spectrum = torch.reshape(in_complex_spectrum, shape=(batch * frames, channels, frequency))
        amplitude_spectrum = torch.reshape(in_amplitude_spectrum, shape=(batch * frames, 1, frequency))
        # print("Complex Spectrum", complex_spectrum.shape)
        # with profiler.record_function("Full band encoder"):
        full_band_encode_outs, global_feature = self.full_band_encoder(complex_spectrum)
        # with profiler.record_function("Sub band encoder"):
        sub_band_encode_outs, local_feature = self.sub_band_encoder(amplitude_spectrum)
        # print(f"FBE out:", full_band_encode_outs.shape, "SBE out:", sub_band_encode_outs.shape)
        # print(global_feature.shape, local_feature.shape)
        merge_feature = torch.cat(tensors=[global_feature, local_feature], dim=2)  # feature cat
        # print(f"Merge layer in:", merge_feature.shape)
        # merge_feature = self.feature_merge_layer(merge_feature)
        # print(f"Merge layer out:", merge_feature.shape)
        # (batch*frames, channels, frequency) -> (batch*frames, channels//2, frequency//2)
        # print(merge_feature.shape)
        _, channels, frequency = merge_feature.shape
        # print(global_feature.shape, local_feature.shape)
        merge_feature = torch.reshape(merge_feature, shape=(batch, frames, channels, frequency))
        merge_feature = torch.permute(merge_feature, dims=(0, 3, 1, 2)).contiguous()
        # (batch, frequency, frames, channels)
        # with profiler.record_function("RNN layer"):
        out_hidden_state = list()
        # print(f"RNN in:", merge_feature.shape)
        for idx, rnn_layer in enumerate(self.dual_path_extension_rnn_list):
            merge_feature, state = rnn_layer(merge_feature, hidden_state[idx])
            out_hidden_state.append(state)
        # print(f"RNN out:", merge_feature.shape)
        merge_feature = torch.permute(merge_feature, dims=(0, 2, 3, 1)).contiguous()
        merge_feature = torch.reshape(merge_feature, shape=(batch * frames, channels, frequency))
        # print(f"Split layer in:", merge_feature.shape)
        # split_feature = self.feature_split_layer(merge_feature)
        first_dim, channels, frequency = merge_feature.shape
        split_feature = torch.reshape(merge_feature, shape=(first_dim, channels, -1, 2))
        # print(f"Split layer out:", split_feature.shape)
        # print(f"FBD in:", split_feature[..., 0].shape,)# full_band_encode_outs.shape)
        # print(f"SBD in:", split_feature[..., 1].shape,)# sub_band_encode_outs.shape)
        # with profiler.record_function("Full band decoder"):
        # print(split_feature[..., 0].shape)
        full_band_mask = self.full_band_decoder(split_feature[..., 0], full_band_encode_outs)
        # with profiler.record_function("Sub band decoder"):
        sub_band_mask = self.sub_band_decoder(split_feature[..., 1], sub_band_encode_outs)

        full_band_mask = torch.reshape(full_band_mask, shape=(batch, frames, 2, -1))
        sub_band_mask = torch.reshape(sub_band_mask, shape=(batch, frames, 1, -1))

        # Zero padding in the DC signal part removes the DC component
        # with profiler.record_function("Mask padding"):
        if self.need_mask:
            full_band_mask = self.mask_padding(full_band_mask) # uncomment for all modeles except TrainConfig48kHzEnc2x_ver1
        sub_band_mask = self.mask_padding(sub_band_mask)
        # print(in_complex_spectrum.shape, full_band_mask.shape)
        full_band_out = in_complex_spectrum * full_band_mask
        sub_band_out = in_amplitude_spectrum * sub_band_mask
        # outputs is (batch, frames, 2, frequency), complex style.

        full_band_out[:, :, 0:1, :] = (full_band_out[:, :, 0:1, :] + sub_band_out) / 2

        if self.print_mask:
            self.print_spec(full_band_mask, sub_band_mask, full_band_out, sub_band_out)

        return full_band_out, out_hidden_state
    

    
class FullSubPathExtension_ver3(nn.Module):
    def __init__(self, configs: TrainConfig_explicit, need_mask: bool = True):
        super().__init__()
        self.full_band_encoder = FullBandEncoder(configs)
        # self.sub_band_encoder = SubBandEncoder(configs) # SubBandEncoder_ver2(configs)
        self.sub_band_encoder = SubBandEncoder_ver2(configs)
        self.num_rnn_modules = configs.dual_path_extension["num_modules"]

        merge_split = configs.merge_split
        merge_channels = merge_split["channels"]
        merge_bands = merge_split["bands"]
        compress_rate = merge_split["compress_rate"]

        print(configs.all_bands, merge_split["channels"] // 2)
        self.features_matching_in = nn.Linear(in_features=configs.all_bands, out_features=merge_split["channels"] // 2)
        self.features_matching_out = nn.Linear(in_features=merge_split["channels"] // 2, out_features=configs.all_bands)

        self.feature_merge_layer = nn.Sequential(
            nn.Linear(in_features=merge_channels, out_features=merge_channels//compress_rate),
            nn.ELU(),
            nn.Conv1d(in_channels=merge_bands, out_channels=merge_bands//compress_rate, kernel_size=1, stride=1)
        )

        # with profiler.record_function("Create GRU"):
        self.dual_path_extension_rnn_list = nn.ModuleList()
        for _ in range(configs.dual_path_extension["num_modules"]):
            self.dual_path_extension_rnn_list.append(DualPathExtensionRNN(**configs.dual_path_extension["parameters"]))

        self.feature_split_layer = nn.Sequential(
            nn.Conv1d(in_channels=merge_bands//compress_rate, out_channels=merge_bands, kernel_size=1, stride=1),
            nn.Linear(in_features=merge_channels//compress_rate, out_features=merge_channels),
            nn.ELU()
        )

        self.full_band_decoder = FullBandDecoder(configs)
        # self.sub_band_decoder = SubBandDecoder(configs) # SubBandDecoder_ver2(configs)
        self.sub_band_decoder = SubBandDecoder_ver2(configs)

        self.mask_padding = nn.ConstantPad2d(padding=(1, 0, 0, 0), value=0.0)
        self.need_mask = need_mask

    def forward(self, in_complex_spectrum: Tensor, in_amplitude_spectrum: Tensor, hidden_state: list):
        """
        :param in_amplitude_spectrum: (batch, frames, 1, frequency)
        :param hidden_state:
        :param in_complex_spectrum: (batch, frames, channels, frequency)
        :return:
        """
        batch, frames, channels, frequency = in_complex_spectrum.shape
        # 16 // 8 for trainconfig
        # with profiler.record_function("Hidden state gen"):
        # hidden_state = [[torch.randn(1, batch * 32, 16 // 8, device=in_complex_spectrum.device) for _ in range(8)] for _ in range(self.num_rnn_modules)] # for rnn2 batch * 32 // 2
        complex_spectrum = torch.reshape(in_complex_spectrum, shape=(batch * frames, channels, frequency))
        amplitude_spectrum = torch.reshape(in_amplitude_spectrum, shape=(batch*frames, 1, frequency))
        # print("Complex Spectrum", complex_spectrum.shape)
        # with profiler.record_function("Full band encoder"):
        full_band_encode_outs, global_feature = self.full_band_encoder(complex_spectrum)
        # with profiler.record_function("Sub band encoder"):
        assert torch.isnan(global_feature).any().item() is False, f"full_band_encoder out has NaNs"
        sub_band_encode_outs, local_feature = self.sub_band_encoder(amplitude_spectrum)
        assert torch.isnan(local_feature).any().item() is False, f"sub_band_encoder out has NaNs"
        # print(f"FBE out:", full_band_encode_outs.shape, "SBE out:", sub_band_encode_outs.shape)
        # print(global_feature.shape, local_feature.shape)
        local_feature = self.features_matching_in(local_feature)
        assert torch.isnan(local_feature).any().item() is False, f"features_matching_in out has NaNs"
        # print(global_feature.shape, local_feature.shape)
        merge_feature = torch.cat(tensors=[global_feature, local_feature], dim=2)  # feature cat
        # print(f"Merge layer in:", merge_feature.shape)
        merge_feature = self.feature_merge_layer(merge_feature)
        assert torch.isnan(merge_feature).any().item() is False, f"feature_merge_layer out has NaNs"
        # print(f"Merge layer out:", merge_feature.shape)
        # (batch*frames, channels, frequency) -> (batch*frames, channels//2, frequency//2)
        _, channels, frequency = merge_feature.shape
        merge_feature = torch.reshape(merge_feature, shape=(batch, frames, channels, frequency))
        merge_feature = torch.permute(merge_feature, dims=(0, 3, 1, 2)).contiguous()
        # (batch, frequency, frames, channels)
        # with profiler.record_function("RNN layer"):
        out_hidden_state = list()
        # print(f"RNN in:", merge_feature.shape)
        for idx, rnn_layer in enumerate(self.dual_path_extension_rnn_list):
            merge_feature, state = rnn_layer(merge_feature, hidden_state[idx])
            out_hidden_state.append(state)
        assert torch.isnan(merge_feature).any().item() is False, f"dual_path_extension_rnn_list out has NaNs"
        # print(f"RNN out:", merge_feature.shape)
        merge_feature = torch.permute(merge_feature, dims=(0, 2, 3, 1)).contiguous()
        merge_feature = torch.reshape(merge_feature, shape=(batch * frames, channels, frequency))
        # print(f"Split layer in:", merge_feature.shape)
        split_feature = self.feature_split_layer(merge_feature)
        assert torch.isnan(split_feature).any().item() is False, f"feature_split_layer out has NaNs"
        first_dim, channels, frequency = split_feature.shape
        split_feature = torch.reshape(split_feature, shape=(first_dim, channels, -1, 2))
        # print(f"Split layer out:", split_feature.shape)
        # print(f"FBD in:", split_feature[..., 0].shape,)# full_band_encode_outs.shape)
        # print(f"SBD in:", split_feature[..., 1].shape,)# sub_band_encode_outs.shape)
        # with profiler.record_function("Full band decoder"):
        # print(split_feature[..., 0].shape)
        full_band_mask = self.full_band_decoder(split_feature[..., 0], full_band_encode_outs)
        assert torch.isnan(full_band_mask).any().item() is False, f"full_band_decoder out has NaNs"
        # with profiler.record_function("Sub band decoder"):
        sub_band_decoder_in = self.features_matching_out(split_feature[..., 1])
        assert torch.isnan(sub_band_decoder_in).any().item() is False, f"features_matching_out out has NaNs"
        sub_band_mask = self.sub_band_decoder(sub_band_decoder_in, sub_band_encode_outs)
        assert torch.isnan(sub_band_mask).any().item() is False, f"sub_band_decoder out has NaNs"

        full_band_mask = torch.reshape(full_band_mask, shape=(batch, frames, 2, -1))
        sub_band_mask = torch.reshape(sub_band_mask, shape=(batch, frames, 1, -1))

        # Zero padding in the DC signal part removes the DC component
        # with profiler.record_function("Mask padding"):
        if self.need_mask:
            full_band_mask = self.mask_padding(full_band_mask) # uncomment for all modeles except TrainConfig48kHzEnc2x_ver1
        sub_band_mask = self.mask_padding(sub_band_mask)
        # print(in_complex_spectrum.shape, full_band_mask.shape)
        full_band_out = in_complex_spectrum * full_band_mask
        assert torch.isnan(full_band_out).any().item() is False, f"full_band_out has NaNs"
        sub_band_out = in_amplitude_spectrum * sub_band_mask
        assert torch.isnan(sub_band_out).any().item() is False, f"sub_band_out has NaNs"
        # outputs is (batch, frames, 2, frequency), complex style.

        full_band_out[:, :, 0:1, :] = (full_band_out[:, :, 0:1, :] + sub_band_out) / 2
        assert torch.isnan(full_band_out[:, :, 0:1, :]).any().item() is False, f"full_band_out[:, :, 0:1, :] has NaNs"
        return full_band_out, out_hidden_state
    

class FullSubPathExtension_ver3_unfold(nn.Module):
    def __init__(self, configs: TrainConfig_explicit, need_mask: bool = True, last_signoid: bool = False):
        super().__init__()
        self.full_band_encoder = FullBandEncoder(configs)
        # self.sub_band_encoder = SubBandEncoder(configs) # SubBandEncoder_ver2(configs)
        self.sub_band_encoder = SubBandEncoder_ver3(configs)
        self.num_rnn_modules = configs.dual_path_extension["num_modules"]

        merge_split = configs.merge_split
        merge_channels = merge_split["channels"]
        merge_bands = merge_split["bands"]
        compress_rate = merge_split["compress_rate"]

        print(configs.all_bands, merge_split["channels"] // 2)
        # self.features_matching_in = nn.Linear(in_features=configs.all_bands, out_features=merge_split["channels"] // 2)
        # self.features_matching_out = nn.Linear(in_features=merge_split["channels"] // 2, out_features=configs.all_bands)

        self.feature_merge_layer = nn.Sequential(
            nn.Linear(in_features=merge_channels, out_features=merge_channels//compress_rate),
            nn.ELU(),
            nn.Conv1d(in_channels=merge_bands, out_channels=merge_bands//compress_rate, kernel_size=1, stride=1)
        )

        # with profiler.record_function("Create GRU"):
        self.dual_path_extension_rnn_list = nn.ModuleList()
        for _ in range(configs.dual_path_extension["num_modules"]):
            self.dual_path_extension_rnn_list.append(DualPathExtensionRNN(**configs.dual_path_extension["parameters"]))

        self.feature_split_layer = nn.Sequential(
            nn.Conv1d(in_channels=merge_bands//compress_rate, out_channels=merge_bands, kernel_size=1, stride=1),
            nn.Linear(in_features=merge_channels//compress_rate, out_features=merge_channels),
            nn.ELU()
        )

        self.full_band_decoder = FullBandDecoder_ver2(configs)
        # self.sub_band_decoder = SubBandDecoder(configs) # SubBandDecoder_ver2(configs)
        self.sub_band_decoder = SubBandDecoder_ver3(configs)

        self.mask_padding = nn.ConstantPad2d(padding=(1, 0, 0, 0), value=0.0)
        self.need_mask = need_mask

        self.chunk_size = configs.unfold_size
        self.unfold = nn.Unfold((1, configs.unfold_size), padding=(0, configs.unfold_padding), stride=(1, configs.unfold_step))
        self.fold = nn.Fold((1, configs.n_fft // 2), (1, configs.unfold_size), padding=(0, configs.unfold_padding), stride=(1, configs.unfold_step))
        self.fold_ones = nn.Fold((1, configs.n_fft // 2), (1, configs.unfold_size), padding=(0, configs.unfold_padding), stride=(1, configs.unfold_step))

        # self.last_signoid = last_signoid
        # if self.last_signoid:
        #     self.sigmoid_fb = nn.Sigmoid()
        #     self.sigmoid_sb = nn.Sigmoid()


    @staticmethod
    def print_spec(fb_mask, sb_mask, fb_out, sb_out):
        plt.figure(figsize=(20, 10))
        plt.subplot(2, 2, 1)
        plt.imshow(fb_mask[0, :, 0].detach().permute(1, 0).numpy(), aspect='auto', origin='lower', norm=LogNorm(vmin=1e-3, vmax=1))
        plt.title("Full Band Mask")
        plt.colorbar()
        plt.subplot(2, 2, 2)
        plt.imshow(sb_mask[0, :, 0].detach().permute(1, 0).numpy(), aspect='auto', origin='lower', norm=LogNorm(vmin=1e-3, vmax=1))
        plt.title("Sub Band Mask")
        plt.colorbar()
        plt.subplot(2, 2, 3)
        plt.imshow(fb_out[0, :, 0].detach().permute(1, 0).numpy(), aspect='auto', origin='lower', norm=LogNorm(vmin=1e-3, vmax=1))
        plt.title("Full Band Masked Output")
        plt.colorbar()
        plt.subplot(2, 2, 4)
        plt.imshow(sb_out[0, :, 0].detach().permute(1, 0).numpy(), aspect='auto', origin='lower', norm=LogNorm(vmin=1e-3, vmax=1))
        plt.title("Sub Band Masked Output")
        plt.colorbar()

    def forward(self, in_complex_spectrum: Tensor, in_amplitude_spectrum: Tensor, hidden_state: list):
        """
        :param in_amplitude_spectrum: (batch, frames, 1, frequency)
        :param hidden_state:
        :param in_complex_spectrum: (batch, frames, channels, frequency)
        :return:
        """
        batch, frames, channels, frequency = in_complex_spectrum.shape
        # 16 // 8 for trainconfig
        # with profiler.record_function("Hidden state gen"):
        # hidden_state = [[torch.randn(1, batch * 32, 16 // 8, device=in_complex_spectrum.device) for _ in range(8)] for _ in range(self.num_rnn_modules)] # for rnn2 batch * 32 // 2
        complex_spectrum = torch.reshape(in_complex_spectrum, shape=(batch * frames, channels, frequency))
        amplitude_spectrum = torch.reshape(in_amplitude_spectrum, shape=(batch*frames, 1, frequency))

        chunked_amplitude_spectrum = self.unfold(amplitude_spectrum)
        # print(chunked_amplitude_spectrum.shape)
        # print(chunked_amplitude_spectrum.shape)
        chunked_amplitude_spectrum = torch.reshape(chunked_amplitude_spectrum, shape=(batch * frames, self.chunk_size, -1, 1))
        chunked_amplitude_spectrum = chunked_amplitude_spectrum.permute(0, 2, 3, 1)
        
        # print("Complex Spectrum", complex_spectrum.shape)
        # with profiler.record_function("Full band encoder"):
        full_band_encode_outs, global_feature = self.full_band_encoder(complex_spectrum)
        # with profiler.record_function("Sub band encoder"):
        assert torch.isnan(global_feature).any().item() is False, f"full_band_encoder out has NaNs"
        # print(chunked_amplitude_spectrum.shape, amplitude_spectrum.shape)
        sub_band_encode_outs, local_feature = self.sub_band_encoder(chunked_amplitude_spectrum)
        assert torch.isnan(local_feature).any().item() is False, f"sub_band_encoder out has NaNs"
        # print(f"FBE out:", full_band_encode_outs.shape, "SBE out:", sub_band_encode_outs.shape)
        # print(global_feature.shape, local_feature.shape)
        # local_feature = self.features_matching_in(local_feature)
        # assert torch.isnan(local_feature).any().item() is False, f"features_matching_in out has NaNs"
        # print(global_feature.shape, local_feature.shape)
        merge_feature = torch.cat(tensors=[global_feature, local_feature], dim=2)  # feature cat
        # print(f"Merge layer in:", merge_feature.shape)
        merge_feature = self.feature_merge_layer(merge_feature)
        assert torch.isnan(merge_feature).any().item() is False, f"feature_merge_layer out has NaNs"
        # print(f"Merge layer out:", merge_feature.shape)
        # (batch*frames, channels, frequency) -> (batch*frames, channels//2, frequency//2)
        _, channels, frequency = merge_feature.shape
        merge_feature = torch.reshape(merge_feature, shape=(batch, frames, channels, frequency))
        merge_feature = torch.permute(merge_feature, dims=(0, 3, 1, 2)).contiguous()
        # (batch, frequency, frames, channels)
        # with profiler.record_function("RNN layer"):
        out_hidden_state = list()
        # print(f"RNN in:", merge_feature.shape)
        for idx, rnn_layer in enumerate(self.dual_path_extension_rnn_list):
            merge_feature, state = rnn_layer(merge_feature, hidden_state[idx])
            out_hidden_state.append(state)
        assert torch.isnan(merge_feature).any().item() is False, f"dual_path_extension_rnn_list out has NaNs"
        # print(f"RNN out:", merge_feature.shape)
        merge_feature = torch.permute(merge_feature, dims=(0, 2, 3, 1)).contiguous()
        merge_feature = torch.reshape(merge_feature, shape=(batch * frames, channels, frequency))
        # print(f"Split layer in:", merge_feature.shape)
        split_feature = self.feature_split_layer(merge_feature)
        assert torch.isnan(split_feature).any().item() is False, f"feature_split_layer out has NaNs"
        first_dim, channels, frequency = split_feature.shape
        split_feature = torch.reshape(split_feature, shape=(first_dim, channels, -1, 2))
        # print(f"Split layer out:", split_feature.shape)
        # print(f"FBD in:", split_feature[..., 0].shape,)# full_band_encode_outs.shape)
        # print(f"SBD in:", split_feature[..., 1].shape,)# sub_band_encode_outs.shape)
        # with profiler.record_function("Full band decoder"):
        # print(split_feature[..., 0].shape)
        full_band_mask = self.full_band_decoder(split_feature[..., 0], full_band_encode_outs)
        assert torch.isnan(full_band_mask).any().item() is False, f"full_band_decoder out has NaNs"
        # with profiler.record_function("Sub band decoder"):
        # sub_band_decoder_in = self.features_matching_out(split_feature[..., 1])
        # assert torch.isnan(sub_band_decoder_in).any().item() is False, f"features_matching_out out has NaNs"
        sub_band_mask = self.sub_band_decoder(split_feature[..., 1], sub_band_encode_outs)
        assert torch.isnan(sub_band_mask).any().item() is False, f"sub_band_decoder out has NaNs"

        full_band_mask = torch.reshape(full_band_mask, shape=(batch, frames, 2, -1))
        # print(sub_band_mask.shape)
        sub_band_mask = sub_band_mask.reshape(batch * frames, self.chunk_size, -1)
        # print(sub_band_mask.shape)

        # if not self.training:
        ones = torch.ones_like(sub_band_mask)

        norm_map = self.fold_ones(ones)

        sub_band_mask = self.fold(sub_band_mask)

        sub_band_mask = sub_band_mask / (norm_map + 1e-8)
        # else:
        #     sub_band_mask = self.fold(sub_band_mask)

        sub_band_mask = torch.reshape(sub_band_mask, shape=(batch, frames, 1, -1))

        # Zero padding in the DC signal part removes the DC component
        # with profiler.record_function("Mask padding"):
        if self.need_mask:
            full_band_mask = self.mask_padding(full_band_mask) # uncomment for all modeles except TrainConfig48kHzEnc2x_ver1
        sub_band_mask = self.mask_padding(sub_band_mask)

        # if self.last_signoid:
        #     full_band_mask_abs, full_band_mask_pha = full_band_mask[:, :, 0:1, :], full_band_mask[:, :, 1:2, :]
        #     full_band_mask_abs = self.sigmoid_fb(full_band_mask_abs)
        #     full_band_mask = torch.cat([full_band_mask_abs, full_band_mask_pha], dim=2)
        #     sub_band_mask = self.sigmoid_sb(sub_band_mask)

        # print(in_complex_spectrum.shape, full_band_mask.shape)
        full_band_out = in_complex_spectrum * full_band_mask
        assert torch.isnan(full_band_out).any().item() is False, f"full_band_out has NaNs"
        sub_band_out = in_amplitude_spectrum * sub_band_mask
        assert torch.isnan(sub_band_out).any().item() is False, f"sub_band_out has NaNs"
        # outputs is (batch, frames, 2, frequency), complex style.

        full_band_out[:, :, 0:1, :] = (full_band_out[:, :, 0:1, :] + sub_band_out) / 2
        # print("Greater than 1.0 in fb:", (full_band_mask[:, :, 0:1, :] > 1.0).any(), full_band_mask[:, :, 0:1, :22].numel(), len(list([x.item() for x in full_band_mask[:, :, 0:1, :22].flatten() if x > 1.])))
        # print("Greater than 1.0 in sb:", (sub_band_mask > 1.0).any(), sub_band_mask[:, :, 0:1, :22].numel(), len(list([x.item() for x in sub_band_mask[:, :, 0:1, :22].flatten() if x > 1.])))
        assert torch.isnan(full_band_out[:, :, 0:1, :]).any().item() is False, f"full_band_out[:, :, 0:1, :] has NaNs"
        return full_band_out, out_hidden_state
    

class FullSubPathExtension_ver3_unfold_light(nn.Module):
    def __init__(self, configs: TrainConfig_explicit, need_mask: bool = True, last_signoid: bool = False):
        super().__init__()
        self.full_band_encoder = FullBandEncoder(configs)
        # self.sub_band_encoder = SubBandEncoder(configs) # SubBandEncoder_ver2(configs)
        self.sub_band_encoder = SubBandEncoder_ver3(configs)
        self.num_rnn_modules = configs.dual_path_extension["num_modules"]

        merge_split = configs.merge_split
        merge_channels = merge_split["channels"]
        merge_bands = merge_split["bands"]
        compress_rate = merge_split["compress_rate"]

        print(configs.all_bands, merge_split["channels"] // 2)
        # self.features_matching_in = nn.Linear(in_features=configs.all_bands, out_features=merge_split["channels"] // 2)
        # self.features_matching_out = nn.Linear(in_features=merge_split["channels"] // 2, out_features=configs.all_bands)

        self.feature_merge_layer = nn.Sequential(
            nn.Linear(in_features=merge_channels, out_features=merge_channels//compress_rate),
            nn.ELU(),
            nn.Conv1d(in_channels=merge_bands, out_channels=merge_bands//compress_rate, kernel_size=1, stride=1)
        )

        # with profiler.record_function("Create GRU"):
        self.dual_path_extension_rnn_list = nn.ModuleList()
        for _ in range(configs.dual_path_extension["num_modules"]):
            self.dual_path_extension_rnn_list.append(DualPathExtensionRNNLight(**configs.dual_path_extension["parameters"]))

        self.feature_split_layer = nn.Sequential(
            nn.Conv1d(in_channels=merge_bands//compress_rate, out_channels=merge_bands, kernel_size=1, stride=1),
            nn.Linear(in_features=merge_channels//compress_rate, out_features=merge_channels),
            nn.ELU()
        )

        self.full_band_decoder = FullBandDecoder_ver2(configs)
        # self.sub_band_decoder = SubBandDecoder(configs) # SubBandDecoder_ver2(configs)
        self.sub_band_decoder = SubBandDecoder_ver3(configs)

        self.mask_padding = nn.ConstantPad2d(padding=(1, 0, 0, 0), value=0.0)
        self.need_mask = need_mask

        self.chunk_size = configs.unfold_size
        self.unfold = nn.Unfold((1, configs.unfold_size), padding=(0, configs.unfold_padding), stride=(1, configs.unfold_step))
        self.fold = nn.Fold((1, configs.n_fft // 2), (1, configs.unfold_size), padding=(0, configs.unfold_padding), stride=(1, configs.unfold_step))
        self.fold_ones = nn.Fold((1, configs.n_fft // 2), (1, configs.unfold_size), padding=(0, configs.unfold_padding), stride=(1, configs.unfold_step))

        # self.last_signoid = last_signoid
        # if self.last_signoid:
        #     self.sigmoid_fb = nn.Sigmoid()
        #     self.sigmoid_sb = nn.Sigmoid()


    @staticmethod
    def print_spec(fb_mask, sb_mask, fb_out, sb_out):
        plt.figure(figsize=(20, 10))
        plt.subplot(2, 2, 1)
        plt.imshow(fb_mask[0, :, 0].detach().permute(1, 0).numpy(), aspect='auto', origin='lower', norm=LogNorm(vmin=1e-3, vmax=1))
        plt.title("Full Band Mask")
        plt.colorbar()
        plt.subplot(2, 2, 2)
        plt.imshow(sb_mask[0, :, 0].detach().permute(1, 0).numpy(), aspect='auto', origin='lower', norm=LogNorm(vmin=1e-3, vmax=1))
        plt.title("Sub Band Mask")
        plt.colorbar()
        plt.subplot(2, 2, 3)
        plt.imshow(fb_out[0, :, 0].detach().permute(1, 0).numpy(), aspect='auto', origin='lower', norm=LogNorm(vmin=1e-3, vmax=1))
        plt.title("Full Band Masked Output")
        plt.colorbar()
        plt.subplot(2, 2, 4)
        plt.imshow(sb_out[0, :, 0].detach().permute(1, 0).numpy(), aspect='auto', origin='lower', norm=LogNorm(vmin=1e-3, vmax=1))
        plt.title("Sub Band Masked Output")
        plt.colorbar()

    def forward(self, in_complex_spectrum: Tensor, in_amplitude_spectrum: Tensor, hidden_state: list):
        """
        :param in_amplitude_spectrum: (batch, frames, 1, frequency)
        :param hidden_state:
        :param in_complex_spectrum: (batch, frames, channels, frequency)
        :return:
        """
        batch, frames, channels, frequency = in_complex_spectrum.shape
        # 16 // 8 for trainconfig
        # with profiler.record_function("Hidden state gen"):
        # hidden_state = [[torch.randn(1, batch * 32, 16 // 8, device=in_complex_spectrum.device) for _ in range(8)] for _ in range(self.num_rnn_modules)] # for rnn2 batch * 32 // 2
        complex_spectrum = torch.reshape(in_complex_spectrum, shape=(batch * frames, channels, frequency))
        amplitude_spectrum = torch.reshape(in_amplitude_spectrum, shape=(batch*frames, 1, frequency))

        chunked_amplitude_spectrum = self.unfold(amplitude_spectrum)
        # print(chunked_amplitude_spectrum.shape)
        # print(chunked_amplitude_spectrum.shape)
        chunked_amplitude_spectrum = torch.reshape(chunked_amplitude_spectrum, shape=(batch * frames, self.chunk_size, -1, 1))
        chunked_amplitude_spectrum = chunked_amplitude_spectrum.permute(0, 2, 3, 1)
        
        # print("Complex Spectrum", complex_spectrum.shape)
        # with profiler.record_function("Full band encoder"):
        full_band_encode_outs, global_feature = self.full_band_encoder(complex_spectrum)
        # with profiler.record_function("Sub band encoder"):
        assert torch.isnan(global_feature).any().item() is False, f"full_band_encoder out has NaNs"
        # print(chunked_amplitude_spectrum.shape, amplitude_spectrum.shape)
        sub_band_encode_outs, local_feature = self.sub_band_encoder(chunked_amplitude_spectrum)
        assert torch.isnan(local_feature).any().item() is False, f"sub_band_encoder out has NaNs"
        # print(f"FBE out:", full_band_encode_outs.shape, "SBE out:", sub_band_encode_outs.shape)
        # print(global_feature.shape, local_feature.shape)
        # local_feature = self.features_matching_in(local_feature)
        # assert torch.isnan(local_feature).any().item() is False, f"features_matching_in out has NaNs"
        # print(global_feature.shape, local_feature.shape)
        merge_feature = torch.cat(tensors=[global_feature, local_feature], dim=2)  # feature cat
        # print(f"Merge layer in:", merge_feature.shape)
        merge_feature = self.feature_merge_layer(merge_feature)
        assert torch.isnan(merge_feature).any().item() is False, f"feature_merge_layer out has NaNs"
        # print(f"Merge layer out:", merge_feature.shape)
        # (batch*frames, channels, frequency) -> (batch*frames, channels//2, frequency//2)
        _, channels, frequency = merge_feature.shape
        merge_feature = torch.reshape(merge_feature, shape=(batch, frames, channels, frequency))
        merge_feature = torch.permute(merge_feature, dims=(0, 3, 1, 2)).contiguous()
        # (batch, frequency, frames, channels)
        # with profiler.record_function("RNN layer"):
        out_hidden_state = list()
        # print(f"RNN in:", merge_feature.shape)
        for idx, rnn_layer in enumerate(self.dual_path_extension_rnn_list):
            merge_feature, state = rnn_layer(merge_feature, hidden_state[idx])
            out_hidden_state.append(state)
        assert torch.isnan(merge_feature).any().item() is False, f"dual_path_extension_rnn_list out has NaNs"
        # print(f"RNN out:", merge_feature.shape)
        merge_feature = torch.permute(merge_feature, dims=(0, 2, 3, 1)).contiguous()
        merge_feature = torch.reshape(merge_feature, shape=(batch * frames, channels, frequency))
        # print(f"Split layer in:", merge_feature.shape)
        split_feature = self.feature_split_layer(merge_feature)
        assert torch.isnan(split_feature).any().item() is False, f"feature_split_layer out has NaNs"
        first_dim, channels, frequency = split_feature.shape
        split_feature = torch.reshape(split_feature, shape=(first_dim, channels, -1, 2))
        # print(f"Split layer out:", split_feature.shape)
        # print(f"FBD in:", split_feature[..., 0].shape,)# full_band_encode_outs.shape)
        # print(f"SBD in:", split_feature[..., 1].shape,)# sub_band_encode_outs.shape)
        # with profiler.record_function("Full band decoder"):
        # print(split_feature[..., 0].shape)
        full_band_mask = self.full_band_decoder(split_feature[..., 0], full_band_encode_outs)
        assert torch.isnan(full_band_mask).any().item() is False, f"full_band_decoder out has NaNs"
        # with profiler.record_function("Sub band decoder"):
        # sub_band_decoder_in = self.features_matching_out(split_feature[..., 1])
        # assert torch.isnan(sub_band_decoder_in).any().item() is False, f"features_matching_out out has NaNs"
        sub_band_mask = self.sub_band_decoder(split_feature[..., 1], sub_band_encode_outs)
        assert torch.isnan(sub_band_mask).any().item() is False, f"sub_band_decoder out has NaNs"

        full_band_mask = torch.reshape(full_band_mask, shape=(batch, frames, 2, -1))
        # print(sub_band_mask.shape)
        sub_band_mask = sub_band_mask.reshape(batch * frames, self.chunk_size, -1)
        # print(sub_band_mask.shape)

        # if not self.training:
        ones = torch.ones_like(sub_band_mask)

        norm_map = self.fold_ones(ones)

        sub_band_mask = self.fold(sub_band_mask)

        sub_band_mask = sub_band_mask / (norm_map + 1e-8)
        # else:
        #     sub_band_mask = self.fold(sub_band_mask)

        sub_band_mask = torch.reshape(sub_band_mask, shape=(batch, frames, 1, -1))

        # Zero padding in the DC signal part removes the DC component
        # with profiler.record_function("Mask padding"):
        if self.need_mask:
            full_band_mask = self.mask_padding(full_band_mask) # uncomment for all modeles except TrainConfig48kHzEnc2x_ver1
        sub_band_mask = self.mask_padding(sub_band_mask)

        # if self.last_signoid:
        #     full_band_mask_abs, full_band_mask_pha = full_band_mask[:, :, 0:1, :], full_band_mask[:, :, 1:2, :]
        #     full_band_mask_abs = self.sigmoid_fb(full_band_mask_abs)
        #     full_band_mask = torch.cat([full_band_mask_abs, full_band_mask_pha], dim=2)
        #     sub_band_mask = self.sigmoid_sb(sub_band_mask)

        # print(in_complex_spectrum.shape, full_band_mask.shape)
        full_band_out = in_complex_spectrum * full_band_mask
        assert torch.isnan(full_band_out).any().item() is False, f"full_band_out has NaNs"
        sub_band_out = in_amplitude_spectrum * sub_band_mask
        assert torch.isnan(sub_band_out).any().item() is False, f"sub_band_out has NaNs"
        # outputs is (batch, frames, 2, frequency), complex style.

        full_band_out[:, :, 0:1, :] = (full_band_out[:, :, 0:1, :] + sub_band_out) / 2
        # print("Greater than 1.0 in fb:", (full_band_mask[:, :, 0:1, :] > 1.0).any(), full_band_mask[:, :, 0:1, :22].numel(), len(list([x.item() for x in full_band_mask[:, :, 0:1, :22].flatten() if x > 1.])))
        # print("Greater than 1.0 in sb:", (sub_band_mask > 1.0).any(), sub_band_mask[:, :, 0:1, :22].numel(), len(list([x.item() for x in sub_band_mask[:, :, 0:1, :22].flatten() if x > 1.])))
        assert torch.isnan(full_band_out[:, :, 0:1, :]).any().item() is False, f"full_band_out[:, :, 0:1, :] has NaNs"
        return full_band_out, out_hidden_state
    

class FullSubPathExtension_ver4(nn.Module):
    def __init__(self, configs: TrainConfig_explicit, need_mask: bool = True):
        super().__init__()
        self.full_band_encoder = FullBandEncoder(configs)
        # self.sub_band_encoder = SubBandEncoder(configs) # SubBandEncoder_ver2(configs)
        self.sub_band_encoder = SubBandEncoder_ver2(configs)
        self.num_rnn_modules = configs.dual_path_extension["num_modules"]

        merge_split = configs.merge_split
        merge_channels = merge_split["channels"]
        merge_bands = merge_split["bands"]
        compress_rate = merge_split["compress_rate"]

        feat_match_in_feats = configs.all_bands
        
        # if configs.overlap is not None:
        #     feat_match_in_feats = sum([configs.end_bands[i] for i in range(len(configs.end_bands))])

        self.features_matching_in = nn.Sequential(
            nn.Linear(in_features=configs.all_bands, out_features=merge_split["channels"] // 2),
            nn.ELU(),
            nn.Linear(in_features=merge_split["channels"] // 2, out_features=merge_split["channels"] // 2)
        )

        # print(merge_split["channels"] // 2, configs.all_bands)
        self.features_matching_out = nn.Sequential(
            nn.Linear(in_features=merge_split["channels"] // 2, out_features=configs.all_bands),
            nn.ELU(),
            nn.Linear(in_features=configs.all_bands, out_features=configs.all_bands)
        )

        self.feature_merge_layer = nn.Sequential(
            nn.Linear(in_features=merge_channels, out_features=merge_channels//compress_rate),
            nn.ELU(),
            nn.Conv1d(in_channels=merge_bands, out_channels=merge_bands//compress_rate, kernel_size=1, stride=1)
        )

        # with profiler.record_function("Create GRU"):
        self.dual_path_extension_rnn_list = nn.ModuleList()
        for _ in range(configs.dual_path_extension["num_modules"]):
            self.dual_path_extension_rnn_list.append(DualPathExtensionRNN(**configs.dual_path_extension["parameters"]))

        self.feature_split_layer = nn.Sequential(
            nn.Conv1d(in_channels=merge_bands//compress_rate, out_channels=merge_bands, kernel_size=1, stride=1),
            nn.Linear(in_features=merge_channels//compress_rate, out_features=merge_channels),
            nn.ELU()
        )

        self.full_band_decoder = FullBandDecoder(configs)
        # self.sub_band_decoder = SubBandDecoder(configs) # SubBandDecoder_ver2(configs)
        self.sub_band_decoder = SubBandDecoder_ver2(configs)

        self.mask_padding = nn.ConstantPad2d(padding=(1, 0, 0, 0), value=0.0)
        self.need_mask = need_mask

        self.overlap = configs.overlap
        if self.overlap:
            self.overlap_sub_band_encoder = SubBandEncoder_ver2(configs)
            self.overlap_sub_band_decoder = SubBandDecoder_ver2(configs)


    def forward(self, in_complex_spectrum: Tensor, in_amplitude_spectrum: Tensor, hidden_state: list):
        """
        :param in_amplitude_spectrum: (batch, frames, 1, frequency)
        :param hidden_state:
        :param in_complex_spectrum: (batch, frames, channels, frequency)
        :return:
        """
        batch, frames, channels, frequency = in_complex_spectrum.shape
        # 16 // 8 for trainconfig
        # with profiler.record_function("Hidden state gen"):
        # hidden_state = [[torch.randn(1, batch * 32, 16 // 8, device=in_complex_spectrum.device) for _ in range(8)] for _ in range(self.num_rnn_modules)] # for rnn2 batch * 32 // 2
        complex_spectrum = torch.reshape(in_complex_spectrum, shape=(batch * frames, channels, frequency))
        amplitude_spectrum = torch.reshape(in_amplitude_spectrum, shape=(batch*frames, 1, frequency))
        # print("Complex Spectrum", complex_spectrum.shape)
        # with profiler.record_function("Full band encoder"):
        full_band_encode_outs, global_feature = self.full_band_encoder(complex_spectrum)
        # with profiler.record_function("Sub band encoder"):
        sub_band_encode_outs, local_feature = self.sub_band_encoder(amplitude_spectrum)

        # print(f"FBE out:", full_band_encode_outs.shape, "SBE out:", sub_band_encode_outs.shape)
        # print(global_feature.shape, local_feature.shape)
        local_feature = self.features_matching_in(local_feature)
        # print(global_feature.shape, local_feature.shape)
        merge_feature = torch.cat(tensors=[global_feature, local_feature], dim=2)  # feature cat
        # print(f"Merge layer in:", merge_feature.shape)
        merge_feature = self.feature_merge_layer(merge_feature)
        # print(f"Merge layer out:", merge_feature.shape)
        # (batch*frames, channels, frequency) -> (batch*frames, channels//2, frequency//2)
        _, channels, frequency = merge_feature.shape
        merge_feature = torch.reshape(merge_feature, shape=(batch, frames, channels, frequency))
        merge_feature = torch.permute(merge_feature, dims=(0, 3, 1, 2)).contiguous()
        # (batch, frequency, frames, channels)
        # with profiler.record_function("RNN layer"):
        out_hidden_state = list()
        # print(f"RNN in:", merge_feature.shape)
        for idx, rnn_layer in enumerate(self.dual_path_extension_rnn_list):
            merge_feature, state = rnn_layer(merge_feature, hidden_state[idx])
            out_hidden_state.append(state)
        # print(f"RNN out:", merge_feature.shape)
        merge_feature = torch.permute(merge_feature, dims=(0, 2, 3, 1)).contiguous()
        merge_feature = torch.reshape(merge_feature, shape=(batch * frames, channels, frequency))
        # print(f"Split layer in:", merge_feature.shape)
        split_feature = self.feature_split_layer(merge_feature)
        first_dim, channels, frequency = split_feature.shape
        split_feature = torch.reshape(split_feature, shape=(first_dim, channels, -1, 2))
        # print(f"Split layer out:", split_feature.shape)
        # print(f"FBD in:", split_feature[..., 0].shape,)# full_band_encode_outs.shape)
        # print(f"SBD in:", split_feature[..., 1].shape,)# sub_band_encode_outs.shape)
        # with profiler.record_function("Full band decoder"):
        # print(split_feature[..., 0].shape)
        full_band_mask = self.full_band_decoder(split_feature[..., 0], full_band_encode_outs)
        # with profiler.record_function("Sub band decoder"):
        # print(split_feature[..., 1].shape)
        sub_band_decoder_in = self.features_matching_out(split_feature[..., 1])
        sub_band_mask = self.sub_band_decoder(sub_band_decoder_in, sub_band_encode_outs)

        full_band_mask = torch.reshape(full_band_mask, shape=(batch, frames, 2, -1))
        sub_band_mask = torch.reshape(sub_band_mask, shape=(batch, frames, 1, -1))

        # Zero padding in the DC signal part removes the DC component
        # with profiler.record_function("Mask padding"):
        if self.need_mask:
            full_band_mask = self.mask_padding(full_band_mask) # uncomment for all modeles except TrainConfig48kHzEnc2x_ver1
        sub_band_mask = self.mask_padding(sub_band_mask)
        # print(in_complex_spectrum.shape, full_band_mask.shape)
        full_band_out = in_complex_spectrum * full_band_mask
        sub_band_out = in_amplitude_spectrum * sub_band_mask
        # outputs is (batch, frames, 2, frequency), complex style.

        full_band_out[:, :, 0:1, :] = (full_band_out[:, :, 0:1, :] + sub_band_out) / 2
        return full_band_out, out_hidden_state


class FullSubPathExtension_3_heads(nn.Module):
    def __init__(self, configs: TrainConfig, need_mask: bool = True):
        super().__init__()
        self.full_band_encoder = FullBandEncoder(configs)
        self.sub_band_encoder = SubBandEncoder(configs) # SubBandEncoder_ver2(configs)
        # self.sub_band_encoder = SubBandEncoder_ver2(configs)
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

        # with profiler.record_function("Create GRU"):
        self.dual_path_extension_rnn_list = nn.ModuleList()
        for _ in range(configs.dual_path_extension["num_modules"]):
            self.dual_path_extension_rnn_list.append(DualPathExtensionRNN(**configs.dual_path_extension["parameters"]))

        self.feature_split_layer = nn.Sequential(
            nn.Conv1d(in_channels=merge_bands//compress_rate, out_channels=merge_bands, kernel_size=1, stride=1),
            nn.Linear(in_features=merge_channels//compress_rate, out_features=merge_channels),
            nn.ELU()
        )

        self.full_band_decoder = FullBandDecoder(configs)
        self.sub_band_decoder = SubBandDecoder(configs) # SubBandDecoder_ver2(configs)
        # self.sub_band_decoder = SubBandDecoder_ver2(configs)

        self.mask_padding = nn.ConstantPad2d(padding=(1, 0, 0, 0), value=0.0)
        self.need_mask = need_mask

    def forward(self, in_complex_spectrum: Tensor, in_amplitude_spectrum: Tensor, hidden_state: list):
        """
        :param in_amplitude_spectrum: (batch, frames, 1, frequency)
        :param hidden_state:
        :param in_complex_spectrum: (batch, frames, channels, frequency)
        :return:
        """
        batch, frames, channels, frequency = in_complex_spectrum.shape
        # 16 // 8 for trainconfig
        # with profiler.record_function("Hidden state gen"):
        # hidden_state = [[torch.randn(1, batch * 32, 16 // 8, device=in_complex_spectrum.device) for _ in range(8)] for _ in range(self.num_rnn_modules)] # for rnn2 batch * 32 // 2
        complex_spectrum = torch.reshape(in_complex_spectrum, shape=(batch * frames, channels, frequency))
        amplitude_spectrum = torch.reshape(in_amplitude_spectrum, shape=(batch*frames, 1, frequency))
        # print("Complex Spectrum", complex_spectrum.shape)
        # with profiler.record_function("Full band encoder"):
        full_band_encode_outs, global_feature = self.full_band_encoder(complex_spectrum)
        # with profiler.record_function("Sub band encoder"):
        sub_band_encode_outs, local_feature = self.sub_band_encoder(amplitude_spectrum)
        # print(f"FBE out:", full_band_encode_outs.shape, "SBE out:", sub_band_encode_outs.shape)
        # print(global_feature.shape, local_feature.shape)
        merge_feature = torch.cat(tensors=[global_feature, local_feature], dim=2)  # feature cat
        # print(f"Merge layer in:", merge_feature.shape)
        merge_feature = self.feature_merge_layer(merge_feature)
        # print(f"Merge layer out:", merge_feature.shape)
        # (batch*frames, channels, frequency) -> (batch*frames, channels//2, frequency//2)
        _, channels, frequency = merge_feature.shape
        merge_feature = torch.reshape(merge_feature, shape=(batch, frames, channels, frequency))
        merge_feature = torch.permute(merge_feature, dims=(0, 3, 1, 2)).contiguous()
        # (batch, frequency, frames, channels)
        # with profiler.record_function("RNN layer"):
        out_hidden_state = list()
        # print(f"RNN in:", merge_feature.shape)
        for idx, rnn_layer in enumerate(self.dual_path_extension_rnn_list):
            merge_feature, state = rnn_layer(merge_feature, hidden_state[idx])
            out_hidden_state.append(state)
        # print(f"RNN out:", merge_feature.shape)
        merge_feature = torch.permute(merge_feature, dims=(0, 2, 3, 1)).contiguous()
        merge_feature = torch.reshape(merge_feature, shape=(batch * frames, channels, frequency))
        # print(f"Split layer in:", merge_feature.shape)
        split_feature = self.feature_split_layer(merge_feature)
        first_dim, channels, frequency = split_feature.shape
        split_feature = torch.reshape(split_feature, shape=(first_dim, channels, -1, 2))
        # print(f"Split layer out:", split_feature.shape)
        # print(f"FBD in:", split_feature[..., 0].shape,)# full_band_encode_outs.shape)
        # print(f"SBD in:", split_feature[..., 1].shape,)# sub_band_encode_outs.shape)
        # with profiler.record_function("Full band decoder"):
        # print(split_feature[..., 0].shape)
        full_band_mask = self.full_band_decoder(split_feature[..., 0], full_band_encode_outs)
        # with profiler.record_function("Sub band decoder"):
        sub_band_mask = self.sub_band_decoder(split_feature[..., 1], sub_band_encode_outs)
        
        # print(full_band_mask.shape)
        full_band_mask = torch.reshape(full_band_mask, shape=(batch, frames, 2 * 3, -1))
        sub_band_mask = torch.reshape(sub_band_mask, shape=(batch, frames, 1 * 3, -1))

        # Zero padding in the DC signal part removes the DC component
        # with profiler.record_function("Mask padding"):
        if self.need_mask:
            full_band_mask = self.mask_padding(full_band_mask) # uncomment for all modeles except TrainConfig48kHzEnc2x_ver1
        sub_band_mask = self.mask_padding(sub_band_mask)
        # print(in_complex_spectrum.shape, full_band_mask.shape)
        # print(full_band_mask.shape, )
        in_complex_spectrum = in_complex_spectrum.repeat(1, 1, 3, 1)
        in_amplitude_spectrum = in_amplitude_spectrum.repeat(1, 1, 3, 1)

        full_band_out = in_complex_spectrum * full_band_mask
        sub_band_out = in_amplitude_spectrum * sub_band_mask
        # outputs is (batch, frames, 2, frequency), complex style.

        full_band_out[:, :, 0::2, :] = (full_band_out[:, :, 0::2, :] + sub_band_out) / 2
        return full_band_out, out_hidden_state

class FullPathExtension(nn.Module):
    def __init__(self, configs: TrainConfig, need_mask: bool = True):
        super().__init__()
        self.full_band_encoder = FullBandEncoder(configs)
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

        # with profiler.record_function("Create GRU"):
        self.dual_path_extension_rnn_list = nn.ModuleList()
        for _ in range(configs.dual_path_extension["num_modules"]):
            self.dual_path_extension_rnn_list.append(DualPathExtensionRNN(**configs.dual_path_extension["parameters"]))

        self.feature_split_layer = nn.Sequential(
            nn.Conv1d(in_channels=merge_bands//compress_rate, out_channels=merge_bands, kernel_size=1, stride=1),
            nn.Linear(in_features=merge_channels//compress_rate, out_features=merge_channels),
            nn.ELU()
        )

        self.full_band_decoder = FullBandDecoder(configs)
        self.sub_band_decoder = SubBandDecoder(configs) # SubBandDecoder_ver2(configs)

        self.mask_padding = nn.ConstantPad2d(padding=(1, 0, 0, 0), value=0.0)
        self.need_mask = need_mask

    def forward(self, in_complex_spectrum: Tensor, in_amplitude_spectrum: Tensor, hidden_state: list):
        """
        :param in_amplitude_spectrum: (batch, frames, 1, frequency)
        :param hidden_state:
        :param in_complex_spectrum: (batch, frames, channels, frequency)
        :return:
        """
        batch, frames, channels, frequency = in_complex_spectrum.shape
        # 16 // 8 for trainconfig
        # with profiler.record_function("Hidden state gen"):
        # hidden_state = [[torch.randn(1, batch * 32, 16 // 8, device=in_complex_spectrum.device) for _ in range(8)] for _ in range(self.num_rnn_modules)] # for rnn2 batch * 32 // 2
        complex_spectrum = torch.reshape(in_complex_spectrum, shape=(batch * frames, channels, frequency))
        # print("Complex Spectrum", complex_spectrum.shape)
        # with profiler.record_function("Full band encoder"):
        full_band_encode_outs, global_feature = self.full_band_encoder(complex_spectrum)
        # with profiler.record_function("Sub band encoder"):
        # print(f"FBE out:", full_band_encode_outs.shape, "SBE out:", sub_band_encode_outs.shape)
        # print(global_feature.shape, local_feature.shape)
        # print(f"Merge layer in:", merge_feature.shape)
        merge_feature = self.feature_merge_layer(global_feature)
        # print(f"Merge layer out:", merge_feature.shape)
        # (batch*frames, channels, frequency) -> (batch*frames, channels//2, frequency//2)
        _, channels, frequency = merge_feature.shape
        merge_feature = torch.reshape(merge_feature, shape=(batch, frames, channels, frequency))
        merge_feature = torch.permute(merge_feature, dims=(0, 3, 1, 2)).contiguous()
        # (batch, frequency, frames, channels)
        # with profiler.record_function("RNN layer"):
        out_hidden_state = list()
        # print(f"RNN in:", merge_feature.shape)
        for idx, rnn_layer in enumerate(self.dual_path_extension_rnn_list):
            merge_feature, state = rnn_layer(merge_feature, hidden_state[idx])
            out_hidden_state.append(state)
        # print(f"RNN out:", merge_feature.shape)
        merge_feature = torch.permute(merge_feature, dims=(0, 2, 3, 1)).contiguous()
        merge_feature = torch.reshape(merge_feature, shape=(batch * frames, channels, frequency))
        # print(f"Split layer in:", merge_feature.shape)
        split_feature = self.feature_split_layer(merge_feature)
        first_dim, channels, frequency = split_feature.shape
        split_feature = torch.reshape(split_feature, shape=(first_dim, channels, -1, 1))
        # print(f"Split layer out:", split_feature.shape)
        # print(f"FBD in:", split_feature[..., 0].shape,)# full_band_encode_outs.shape)
        # print(f"SBD in:", split_feature[..., 1].shape,)# sub_band_encode_outs.shape)
        # with profiler.record_function("Full band decoder"):
        # print(split_feature[..., 0].shape)
        full_band_mask = self.full_band_decoder(split_feature[..., 0], full_band_encode_outs)
        # with profiler.record_function("Sub band decoder"):

        full_band_mask = torch.reshape(full_band_mask, shape=(batch, frames, 2, -1))

        # Zero padding in the DC signal part removes the DC component
        # with profiler.record_function("Mask padding"):
        if self.need_mask:
            full_band_mask = self.mask_padding(full_band_mask) # uncomment for all modeles except TrainConfig48kHzEnc2x_ver1
        # print(in_complex_spectrum.shape, full_band_mask.shape)
        full_band_out = in_complex_spectrum * full_band_mask
        # outputs is (batch, frames, 2, frequency), complex style.

        # full_band_out[:, :, 0:1, :] = (full_band_out[:, :, 0:1, :] + sub_band_out) / 2
        return full_band_out, out_hidden_state

class SubPathExtension(nn.Module):
    def __init__(self, configs: TrainConfig, need_mask: bool = True):
        super().__init__()
        self.sub_band_encoder = SubBandEncoder(configs) # SubBandEncoder_ver2(configs)
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

        # with profiler.record_function("Create GRU"):
        self.dual_path_extension_rnn_list = nn.ModuleList()
        for _ in range(configs.dual_path_extension["num_modules"]):
            self.dual_path_extension_rnn_list.append(DualPathExtensionRNN(**configs.dual_path_extension["parameters"]))

        self.feature_split_layer = nn.Sequential(
            nn.Conv1d(in_channels=merge_bands//compress_rate, out_channels=merge_bands, kernel_size=1, stride=1),
            nn.Linear(in_features=merge_channels//compress_rate, out_features=merge_channels),
            nn.ELU()
        )

        self.full_band_decoder = FullBandDecoder(configs)
        self.sub_band_decoder = SubBandDecoder(configs) # SubBandDecoder_ver2(configs)

        self.mask_padding = nn.ConstantPad2d(padding=(1, 0, 0, 0), value=0.0)
        self.need_mask = need_mask

    def forward(self, in_complex_spectrum: Tensor, in_amplitude_spectrum: Tensor, hidden_state: list):
        """
        :param in_amplitude_spectrum: (batch, frames, 1, frequency)
        :param hidden_state:
        :param in_complex_spectrum: (batch, frames, channels, frequency)
        :return:
        """
        batch, frames, channels, frequency = in_complex_spectrum.shape
        # 16 // 8 for trainconfig
        # with profiler.record_function("Hidden state gen"):
        # hidden_state = [[torch.randn(1, batch * 32, 16 // 8, device=in_complex_spectrum.device) for _ in range(8)] for _ in range(self.num_rnn_modules)] # for rnn2 batch * 32 // 2
        complex_spectrum = torch.reshape(in_complex_spectrum, shape=(batch * frames, channels, frequency))
        # amplitude_spectrum = torch.reshape(in_amplitude_spectrum, shape=(batch*frames, 1, frequency))
        # print("Complex Spectrum", complex_spectrum.shape)
        # with profiler.record_function("Full band encoder"):
        # full_band_encode_outs, global_feature = self.full_band_encoder(complex_spectrum)
        # with profiler.record_function("Sub band encoder"):
        sub_band_encode_outs, local_feature = self.sub_band_encoder(complex_spectrum)
        # print(f"FBE out:", full_band_encode_outs.shape, "SBE out:", sub_band_encode_outs.shape)
        # print(global_feature.shape, local_feature.shape)
        # merge_feature = torch.cat(tensors=[global_feature, local_feature], dim=2)  # feature cat
        # print(f"Merge layer in:", merge_feature.shape)
        merge_feature = self.feature_merge_layer(local_feature)
        # print(f"Merge layer out:", merge_feature.shape)
        # (batch*frames, channels, frequency) -> (batch*frames, channels//2, frequency//2)
        _, channels, frequency = merge_feature.shape
        merge_feature = torch.reshape(merge_feature, shape=(batch, frames, channels, frequency))
        merge_feature = torch.permute(merge_feature, dims=(0, 3, 1, 2)).contiguous()
        # (batch, frequency, frames, channels)
        # with profiler.record_function("RNN layer"):
        out_hidden_state = list()
        # print(f"RNN in:", merge_feature.shape)
        for idx, rnn_layer in enumerate(self.dual_path_extension_rnn_list):
            merge_feature, state = rnn_layer(merge_feature, hidden_state[idx])
            out_hidden_state.append(state)
        # print(f"RNN out:", merge_feature.shape)
        merge_feature = torch.permute(merge_feature, dims=(0, 2, 3, 1)).contiguous()
        merge_feature = torch.reshape(merge_feature, shape=(batch * frames, channels, frequency))
        # print(f"Split layer in:", merge_feature.shape)
        split_feature = self.feature_split_layer(merge_feature)
        first_dim, channels, frequency = split_feature.shape
        split_feature = torch.reshape(split_feature, shape=(first_dim, channels, -1, 1))
        # print(f"Split layer out:", split_feature.shape)
        # print(f"FBD in:", split_feature[..., 0].shape,)# full_band_encode_outs.shape)
        # print(f"SBD in:", split_feature[..., 1].shape,)# sub_band_encode_outs.shape)
        # with profiler.record_function("Full band decoder"):
        # print(split_feature[..., 0].shape)
        # full_band_mask = self.full_band_decoder(split_feature[..., 0], full_band_encode_outs)
        # with profiler.record_function("Sub band decoder"):
        sub_band_mask = self.sub_band_decoder(split_feature[..., 0], sub_band_encode_outs)

        # full_band_mask = torch.reshape(full_band_mask, shape=(batch, frames, 2, -1))
        sub_band_mask = torch.reshape(sub_band_mask, shape=(batch, frames, 1, -1))

        # Zero padding in the DC signal part removes the DC component
        # with profiler.record_function("Mask padding"):
        # if self.need_mask:
        #     full_band_mask = self.mask_padding(full_band_mask) # uncomment for all modeles except TrainConfig48kHzEnc2x_ver1
        sub_band_mask = self.mask_padding(sub_band_mask)
        # print(in_complex_spectrum.shape, full_band_mask.shape)
        # full_band_out = in_complex_spectrum * full_band_mask
        sub_band_out = in_amplitude_spectrum * sub_band_mask
        # outputs is (batch, frames, 2, frequency), complex style.

        # full_band_out[:, :, 0:1, :] = (full_band_out[:, :, 0:1, :] + sub_band_out) / 2
        return sub_band_out, out_hidden_state

class ResidualBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, n_layers, activation_class=nn.ReLU, dropout_p=0.3, reverse=False,
                 do_scale=True, change_size=0):
        super(ResidualBlock, self).__init__()

        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.activation_class = activation_class
        self.dropout_p = dropout_p
        self.reverse = reverse
        self.do_scale = do_scale
        self.change_size = change_size

        assert self.kernel_size % 2 == 1, "wrong kernel_size"

        if self.reverse:
            self.conv_class = nn.ConvTranspose2d
            self.scale_class = partial(
                nn.ConvTranspose2d,
                in_channels=self.c_out,
                out_channels=self.c_out,
                stride=2,
            )
            self.conv_name_prefix = "deconv"
            self.scale_name = "upscale"
            self.change_name = "add_size"
        else:
            self.conv_class = nn.Conv2d
            self.scale_class = nn.MaxPool2d
            self.conv_name_prefix = "conv"
            self.scale_name = "downscale"
            self.change_name = "sub_size"

        layers = [
            nn.Sequential(OrderedDict([
                (
                    self.conv_name_prefix + "_1",
                    self.conv_class(
                        in_channels=self.c_in if i == 0 else self.c_out,
                        out_channels=self.c_out,
                        kernel_size=self.kernel_size,
                        padding=(self.kernel_size - 1) // 2,
                        bias=False,
                    ),
                ),
                ("bnorm_1", nn.BatchNorm2d(self.c_out)),
                ("act", self.activation_class()),
                ("drop", nn.Dropout2d(self.dropout_p)),
                (
                    self.conv_name_prefix + "_2",
                    self.conv_class(
                        in_channels=self.c_out,
                        out_channels=self.c_out,
                        kernel_size=self.kernel_size,
                        padding=(self.kernel_size - 1) // 2,
                        bias=False,
                    ),
                ),
                ("bnorm_2", nn.BatchNorm2d(self.c_out)),
            ])) for i in range(self.n_layers)
        ]

        self.layers = nn.Sequential(OrderedDict([(f"layer_{i}", layer) for i, layer in enumerate(layers)]))

        self.adapt_residual = nn.Identity() if self.c_in == self.c_out else self.conv_class(self.c_in, self.c_out, 1)

        scale = [
            (self.change_name,
             self.conv_class(self.c_out, self.c_out, self.change_size + 1) if self.change_size > 0 else nn.Identity()),
            (self.scale_name, self.scale_class(kernel_size=2) if self.do_scale else nn.Identity()),
        ]
        self.scale = nn.Sequential(OrderedDict(scale))

    def forward(self, x):
        out = self.layers(x)
        out = out + self.adapt_residual(x)
        return self.scale(out)


class ConvBlock(nn.Module):

    def __init__(self, c_in, c_out, kernel_size, stride=1, padding=3, out_pad=0, conv=nn.Conv2d,
                 activation_class=nn.ReLU, dropout_p=0.3):
        super(ConvBlock, self).__init__()
        if conv == nn.Conv2d:
            self.conv = conv(in_channels=c_in, out_channels=c_out, kernel_size=kernel_size, stride=stride,
                             padding=padding)
        elif conv == nn.ConvTranspose2d:
            self.conv = conv(in_channels=c_in, out_channels=c_out, kernel_size=kernel_size, stride=stride,
                             padding=padding,
                             output_padding=out_pad)

        self.bn = nn.BatchNorm2d(c_out)
        self.act = activation_class()
        self.drop = nn.Dropout2d(dropout_p)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class DiscriminatorModel(nn.Module):

    def __init__(self, c_in, activation_class=nn.ReLU, dropout_p=0.3):
        super(DiscriminatorModel, self).__init__()
        self.activation_class = activation_class
        self.dropout_p = dropout_p

        self.seq = nn.Sequential(OrderedDict([
            ('conv1', ConvBlock(c_in, 64, kernel_size=4, stride=2, padding=1, activation_class=self.activation_class, dropout_p=self.dropout_p)),
            ('conv2', ConvBlock(64, 128, kernel_size=4, stride=2, padding=1, activation_class=self.activation_class, dropout_p=self.dropout_p)),
            ('conv3', ConvBlock(128, 256, kernel_size=4, stride=2, padding=1, activation_class=self.activation_class, dropout_p=self.dropout_p)),
            ('conv4', ConvBlock(256, 512, kernel_size=4, stride=1, padding=1, activation_class=self.activation_class, dropout_p=self.dropout_p)),
            ('conv5', ConvBlock(512, 2, kernel_size=4, stride=1, padding=1, activation_class=self.activation_class, dropout_p=self.dropout_p)),
            # ('act', nn.Softmax(dim=1))
        ]))

    def forward(self, x):
        batch, frames, channels, frequency = x.shape
        x = x.reshape(batch, channels, frames, frequency)
        x = self.seq(x)
        return x

# Input [(0, 16)]: torch.Size([569, 1, 257])
# Sub spectrum [(0, 16)]: torch.Size([569, 1, 16])
# Output [(0, 16)]: torch.Size([569, 64, 8])

# Input [(16, 34)]: torch.Size([569, 1, 257])
# Sub spectrum [(16, 34)]: torch.Size([569, 1, 18])
# Output [(16, 34)]: torch.Size([569, 64, 6])

# Input [(34, 70)]: torch.Size([569, 1, 257])
# Sub spectrum [(34, 70)]: torch.Size([569, 1, 36])
# Output [(34, 70)]: torch.Size([569, 64, 6])

# Input [(70, 136)]: torch.Size([569, 1, 257])
# Sub spectrum [(70, 136)]: torch.Size([569, 1, 66])
# Output [(70, 136)]: torch.Size([569, 64, 6])

# Input [(136, 257)]: torch.Size([569, 1, 257])
# Sub spectrum [(136, 257)]: torch.Size([569, 1, 121])
# Output [(136, 257)]: torch.Size([569, 64, 6])

if __name__ == '__main__':
    # discriminator = DiscriminatorModel(c_in=2)
    # x = torch.randn((1, 256, 2, 256))
    # y = discriminator(x)
    # print(y.shape)

    # con = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=2, dilation=1, padding=0, bias=False)
    # nn.init.ones_(con.weight)
    # x = torch.tensor([[[1., 2., 3., 4., 5., 6.]]])
    # print(x)
    # print(con(x))
    # tensor([[[1., 3., 5., 7., 9., 5.]]], grad_fn=<ConvolutionBackward0>)

    x1 = torch.randn([1, 569, 2, 257])
    x2 = torch.randn([1, 569, 1, 257])
    configs = TrainConfigLarge1()
    fspen = FullSubPathExtension(configs=configs)
    output, _ = fspen(x1, x2)
    print(output.size())
    # torch.Size([569, 64, 32]) torch.Size([569, 64, 32])

    # output: torch.Size([1, 569, 2, 257])
    # Regular FullBandEnc:
    # torch.Size([569, 4, 128])
    # torch.Size([569, 16, 64])
    # torch.Size([569, 64, 32])
    # Causal FullBandEnc:
    # torch.Size([569, 4, 128])
    # torch.Size([569, 16, 64])
    # torch.Size([569, 64, 32])
    #
    # Causal FBD in: torch.Size([569, 64, 32])
    # Regular FBD in: torch.Size([569, 64, 32])
    # Causal torch.Size([569, 16, 76]) torch.Size([569, 16, 64])
    # Regular
    # decode torch.Size([569, 64, 32]) torch.Size([569, 64, 32])
    # decode torch.Size([569, 16, 64]) torch.Size([569, 16, 64])
    # decode torch.Size([569, 4, 128]) torch.Size([569, 4, 128])
    #
    # Causal FBD in: torch.Size([569, 64, 32])
    # decode torch.Size([569, 64, 32]) torch.Size([569, 64, 32])
    # decode conv out: torch.Size([569, 64, 32])
    # decode convT out: torch.Size([569, 16, 76])
    # decode torch.Size([569, 16, 76]) torch.Size([569, 16, 64])

    # Regular FBD in: torch.Size([569, 64, 32])
    # decode torch.Size([569, 64, 32]) torch.Size([569, 64, 32])
    # decode conv out: torch.Size([569, 64, 32])
    # decode convT out: torch.Size([569, 16, 64])
    # decode torch.Size([569, 16, 64]) torch.Size([569, 16, 64])