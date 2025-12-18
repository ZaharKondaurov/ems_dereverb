import torch

from torch import nn, Tensor
import torch.autograd.profiler as profiler
from models.en_decoder import FullBandEncoderBlock, FullBandDecoderBlock
from models.en_decoder import SubBandEncoderBlock, SubBandDecoderBlock
from models.sequence_modules import DualPathExtensionRNN
from src.fspen_configs import TrainConfig, TrainConfigLarge, TrainConfigLarge1

from functools import partial
from collections import OrderedDict


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
        for encoder in self.full_band_encoder:
            # print(complex_spectrum.size())
            complex_spectrum = encoder(complex_spectrum)
            full_band_encodes.append(complex_spectrum)

        global_feature = self.global_features(complex_spectrum)

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


class FullBandDecoder(nn.Module):
    def __init__(self, configs: TrainConfig):
        super().__init__()
        self.full_band_decoders = nn.ModuleList()
        for decoder_name, parameters in configs.full_band_decoder.items():
            self.full_band_decoders.append(
                FullBandDecoderBlock(**parameters))

    def forward(self, feature: Tensor, encode_outs: list):
        for decoder, encode_out in zip(self.full_band_decoders, encode_outs):
            feature = decoder(feature, encode_out)

        return feature


class SubBandDecoder(nn.Module):
    def __init__(self, configs: TrainConfig):
        super().__init__()
        start_idx = 0
        self.sub_band_decoders = nn.ModuleList()
        for (decoder_name, parameters), bands in zip(configs.sub_band_decoder.items(), configs.bands_num_in_groups):
            end_idx = start_idx + bands
            self.sub_band_decoders.append(SubBandDecoderBlock(start_idx=start_idx, end_idx=end_idx, **parameters))

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


class FullSubPathExtension(nn.Module):
    def __init__(self, configs: TrainConfig, need_mask: bool = True):
        super().__init__()
        self.full_band_encoder = FullBandEncoder(configs)
        self.sub_band_encoder = SubBandEncoder(configs)
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
        self.sub_band_decoder = SubBandDecoder(configs)

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

        full_band_out[:, :, 0:1, :] = (full_band_out[:, :, 0:1, :] + sub_band_out) / 2
        return full_band_out, out_hidden_state


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