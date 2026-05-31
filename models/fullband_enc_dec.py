from models.en_dec_blocks import *
from src.fspen_configs import TrainConfig
from torch import nn, Tensor
import torch

class FullBandEncoder(nn.Module):
    def __init__(self, configs: TrainConfig):
        super().__init__()

        last_channels = 0
        self.full_band_encoder = nn.ModuleList()
        for encoder_name, conv_parameter in configs.full_band_encoder.items():
            self.full_band_encoder.append(FullBandEncoderBlock(**conv_parameter, act=configs.mag_act["full"]))
            last_channels = conv_parameter["out_channels"]

        self.global_features = nn.Conv1d(in_channels=last_channels, out_channels=last_channels, kernel_size=1, stride=1)

    def forward(self, complex_spectrum: Tensor):
        """
        :param complex_spectrum: (batch*frame, channels, frequency)
        :return:
        """
        full_band_encodes = []
        for i, encoder in enumerate(self.full_band_encoder):
            complex_spectrum = encoder(complex_spectrum)
            assert torch.isnan(complex_spectrum).any().item() is False, f"fullband enc conv_{i} out has NaNs"
            full_band_encodes.append(complex_spectrum)

        global_feature = self.global_features(complex_spectrum)
        assert torch.isnan(global_feature).any().item() is False, f"fullband global_feature out has NaNs"

        return full_band_encodes[::-1], global_feature



class FullBandDecoder(nn.Module):
    def __init__(self, configs: TrainConfig):
        super().__init__()
        self.full_band_decoders = nn.ModuleList()
        fbd_items = configs.full_band_decoder.items()
        for ind, (decoder_name, parameters) in enumerate(fbd_items):
            split_act = False
            mag_act = configs.mag_act["full"]
            pha_act = configs.mag_act["full"]
            if ind == len(fbd_items) - 1 and configs.split_last:
                split_act = True
                mag_act = configs.last_act["mag_act"]
                pha_act = configs.last_act["pha_act"]
            self.full_band_decoders.append(FullBandDecoderBlock(**parameters, split_act=split_act, mag_act=mag_act, pha_act=pha_act))

    def forward(self, feature: Tensor, encode_outs: list):
        for i, (decoder, encode_out) in enumerate(zip(self.full_band_decoders, encode_outs)):
            feature = decoder(feature, encode_out)
            assert torch.isnan(feature).any().item() is False, f"fullband dec conv_{i} out has NaNs"

        return feature