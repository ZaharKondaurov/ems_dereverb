from models.en_dec_blocks import *
from src.fspen_configs import TrainConfig
from torch import nn, Tensor
import torch

class SubBandEncoder(nn.Module):
    def __init__(self, configs: TrainConfig):
        super().__init__()

        self.sub_band_encoders = nn.ModuleList()
        for encoder_name, conv_parameters in configs.sub_band_encoder.items():
            self.sub_band_encoders.append(SubBandEncoderBlock(**conv_parameters["conv"], mag_act=configs.mag_act["sub"]))

    def forward(self, amplitude_spectrum: Tensor):
        sub_band_encodes = list()
        for encoder in self.sub_band_encoders:
            encode_out = encoder(amplitude_spectrum)
            sub_band_encodes.append(encode_out)
        local_feature = torch.cat(sub_band_encodes, dim=2)  # feature cat

        return sub_band_encodes, local_feature
    

class SubBandEncoder_ext(nn.Module):
    def __init__(self, configs: TrainConfig):
        super().__init__()

        self.sub_band_encoders = nn.ModuleList()
        self.freq_bounds = []
        for encoder_name, layer_parameters in configs.sub_band_encoder.items():
            self.freq_bounds.append(layer_parameters["bounds"])
            sub_band_layer = nn.ModuleList()
            for conv_parameters in layer_parameters["convs"]:
                sub_band_layer.append(FullBandEncoderBlock(**conv_parameters, normalize=False, act=configs.mag_act["sub"]))

            self.sub_band_encoders.append(sub_band_layer)

    def forward(self, amplitude_spectrum: Tensor):
        sub_band_encodes = list()
        for ind, encoder in enumerate(self.sub_band_encoders):
            start_idx = self.freq_bounds[ind]["start_frequency"]
            end_idx = self.freq_bounds[ind]["end_frequency"]
            
            encode_in = amplitude_spectrum[:, :, start_idx: end_idx]
            conv_outs = []
            for i, conv in enumerate(encoder):
                encode_out = conv(encode_in)
                assert torch.isnan(encode_out).any().item() is False, f"subband enc conv {ind} group {i} layer out has NaNs"

                conv_outs.append(encode_out)
                encode_in = encode_out

            sub_band_encodes.append(conv_outs)
            
        local_feature = torch.cat([outs[-1] for outs in sub_band_encodes], dim=2)  # feature cat

        return sub_band_encodes, local_feature
    

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
        sub_decoder_outs = []
        for decoder, sub_encode in zip(self.sub_band_decoders, sub_encodes):
            sub_decoder_out = decoder(feature, sub_encode)
            sub_decoder_outs.append(sub_decoder_out)

        sub_decoder_outs = torch.cat(tensors=sub_decoder_outs, dim=1)  # feature cat

        return sub_decoder_outs
  

class SubBandDecoder_ext(nn.Module):
    def __init__(self, configs: TrainConfig):
        super().__init__()
        
        start_idx = 0
        self.bands = []
        self.sub_band_decoders = nn.ModuleList()
        sbd_items = configs.sub_band_decoder.items()

        self.overlap = configs.overlap
        for i, (decoder_name, layer_parameters) in enumerate(sbd_items):
            self.bands.append({"start": start_idx, "end": start_idx + layer_parameters["width"]})
            start_idx = start_idx + layer_parameters["width"]
            sub_band_layer = nn.ModuleList()
            mag_act = configs.mag_act["sub"]
            for ind, conv_parameters in enumerate(layer_parameters["convs"]):
                if ind == len(layer_parameters["convs"]) - 1 and configs.split_last:
                    mag_act = configs.last_act["mag_act"]

                sub_band_layer.append(FullBandDecoderBlock(**conv_parameters, normalize=False, split_act=False, mag_act=mag_act))
            
            self.sub_band_decoders.append(sub_band_layer)


    def forward(self, feature: Tensor, sub_encodes: list):
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

