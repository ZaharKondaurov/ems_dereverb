from typing import Dict, Tuple

from torch.nn import Conv1d, ConvTranspose1d

from pydantic import BaseModel, field_validator
from src.causal_convs import CausalConv1d, CausalConvTranspose1d

def get_sub_bands(band_parameters: dict):
    group_bands = list()
    group_band_width = list()
    for key, value in band_parameters.items():
        num_band = (value["group_width"] - value["conv"]["kernel_size"] +
                    2 * value["conv"]["padding"]) // value["conv"]["stride"] + 1
        sub_band_width = value["group_width"] // num_band
        group_bands.append(num_band)
        group_band_width.append(sub_band_width)

    return tuple(group_bands), tuple(group_band_width)


class TrainConfig(BaseModel):
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 256
    train_frames: int = 62
    train_points: int = (train_frames - 1) * hop_length

    full_band_encoder: Dict[str, dict] = {
        "encoder1": {"in_channels": 2, "out_channels": 4, "kernel_size": 6, "stride": 2, "padding": 2},
        "encoder2": {"in_channels": 4, "out_channels": 16, "kernel_size": 8, "stride": 2, "padding": 3},
        "encoder3": {"in_channels": 16, "out_channels": 32, "kernel_size": 6, "stride": 2, "padding": 2}
    }
    full_band_decoder: Dict[str, dict] = {
        "decoder1": {"in_channels": 64, "out_channels": 16, "kernel_size": 6, "stride": 2, "padding": 2},
        "decoder2": {"in_channels": 32, "out_channels": 4, "kernel_size": 8, "stride": 2, "padding": 3},
        "decoder3": {"in_channels": 8, "out_channels": 2, "kernel_size": 6, "stride": 2, "padding": 2}
    }

    sub_band_encoder: Dict[str, dict] = {
        "encoder1": {"group_width": 16, "conv": {"start_frequency": 0, "end_frequency": 16, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 4, "stride": 2, "padding": 1}},
        "encoder2": {"group_width": 18, "conv": {"start_frequency": 16, "end_frequency": 34, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 7, "stride": 3, "padding": 2}},
        "encoder3": {"group_width": 36, "conv": {"start_frequency": 34, "end_frequency": 70, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 11, "stride": 5, "padding": 2}},
        "encoder4": {"group_width": 66, "conv": {"start_frequency": 70, "end_frequency": 136, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 20, "stride": 10, "padding": 4}},
        "encoder5": {"group_width": 121, "conv": {"start_frequency": 136, "end_frequency": 257, "in_channels": 1,
                                                  "out_channels": 32, "kernel_size": 30, "stride": 20, "padding": 5}}
    }
    merge_split: dict = {"channels": 64, "bands": 32, "compress_rate": 2}
    bands_num_in_groups: Tuple[int] = get_sub_bands(sub_band_encoder)[0]
    band_width_in_groups: Tuple[int] = get_sub_bands(sub_band_encoder)[1]

    sub_band_decoder: Dict[str, dict] = {f"decoder{idx}": {"in_features": 64, "out_features": width}
                                         for idx, width in enumerate(band_width_in_groups)}

    dual_path_extension: dict = {
        "num_modules": 3,
        "parameters": {"input_size": 16, "intra_hidden_size": 16, "inter_hidden_size": 16,
                       "groups": 8, "rnn_type": "GRU"}
    }

    @field_validator("sub_band_decoder")
    def sub_band_decoder_validate(cls, decoders):
        for decoder in decoders:
            if decoder["out_feature"] < 2:
                raise ValueError(f"values should > 2, but got {decoder['out_feature']}")
            

class TrainConfig48kHz(BaseModel):
    sample_rate: int = 48_000
    n_fft: int = 1024
    hop_length: int = 512
    train_frames: int = 62
    train_points: int = (train_frames - 1) * hop_length

    full_band_encoder: Dict[str, dict] = {
        "encoder1": {"in_channels": 2, "out_channels": 4, "kernel_size": 6, "stride": 2, "padding": 2},
        "encoder2": {"in_channels": 4, "out_channels": 16, "kernel_size": 8, "stride": 2, "padding": 3},
        "encoder3": {"in_channels": 16, "out_channels": 32, "kernel_size": 6, "stride": 2, "padding": 2},
        # "encoder4": {"in_channels": 16, "out_channels": 32, "kernel_size": 8, "stride": 2, "padding": 2}
    }
    full_band_decoder: Dict[str, dict] = {
        "decoder1": {"in_channels": 64, "out_channels": 16, "kernel_size": 6, "stride": 2, "padding": 2},
        "decoder2": {"in_channels": 32, "out_channels": 4, "kernel_size": 8, "stride": 2, "padding": 3},
        "decoder3": {"in_channels": 8, "out_channels": 2, "kernel_size": 6, "stride": 2, "padding": 2}
    }

    sub_band_encoder: Dict[str, dict] = {
        "encoder1": {"group_width": 32, "conv": {"start_frequency": 0, "end_frequency": 32, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 4, "stride": 2, "padding": 1}},
        "encoder2": {"group_width": 36, "conv": {"start_frequency": 32, "end_frequency": 68, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 7, "stride": 3, "padding": 2}},
        "encoder3": {"group_width": 72, "conv": {"start_frequency": 68, "end_frequency": 140, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 11, "stride": 6, "padding": 3}},
        "encoder4": {"group_width": 132, "conv": {"start_frequency": 140, "end_frequency": 272, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 21, "stride": 10, "padding": 0}},
        "encoder5": {"group_width": 241, "conv": {"start_frequency": 272, "end_frequency": 513, "in_channels": 1,
                                                  "out_channels": 32, "kernel_size": 30, "stride": 20, "padding": 5}}
    }
    merge_split: dict = {"channels": 128, "bands": 32, "compress_rate": 2}
    bands_num_in_groups: Tuple[int] = get_sub_bands(sub_band_encoder)[0]
    band_width_in_groups: Tuple[int] = get_sub_bands(sub_band_encoder)[1]

    sub_band_decoder: Dict[str, dict] = {f"decoder{idx}": {"in_features": 64, "out_features": width}
                                         for idx, width in enumerate(band_width_in_groups)}

    dual_path_extension: dict = {
        "num_modules": 3,
        "parameters": {"input_size": 16, "intra_hidden_size": 16, "inter_hidden_size": 16,
                       "groups": 8, "rnn_type": "GRU"}
    }

    @field_validator("sub_band_decoder")
    def sub_band_decoder_validate(cls, decoders):
        for decoder in decoders:
            if decoder["out_feature"] < 2:
                raise ValueError(f"values should > 2, but got {decoder['out_feature']}")

class TrainConfig48kHzEnc(BaseModel):
    sample_rate: int = 48_000
    n_fft: int = 1024
    hop_length: int = 512
    train_frames: int = 62
    train_points: int = (train_frames - 1) * hop_length

    full_band_encoder: Dict[str, dict] = {
        "encoder1": {"in_channels": 2, "out_channels": 4, "kernel_size": 6, "stride": 2, "padding": 2},
        "encoder2": {"in_channels": 4, "out_channels": 16, "kernel_size": 8, "stride": 2, "padding": 3},
        "encoder3": {"in_channels": 16, "out_channels": 32, "kernel_size": 6, "stride": 2, "padding": 2},
        # "encoder4": {"in_channels": 16, "out_channels": 32, "kernel_size": 8, "stride": 2, "padding": 2}
    }
    full_band_decoder: Dict[str, dict] = {
        "decoder1": {"in_channels": 64, "out_channels": 16, "kernel_size": 6, "stride": 2, "padding": 2},
        "decoder2": {"in_channels": 32, "out_channels": 4, "kernel_size": 8, "stride": 2, "padding": 3},
        "decoder3": {"in_channels": 8, "out_channels": 2, "kernel_size": 6, "stride": 2, "padding": 2}
    }

    sub_band_encoder: Dict[str, dict] = {
        "encoder1": {"group_width": 36, "conv": {"start_frequency": 0, "end_frequency": 36, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 5, "stride": 3, "padding": 1}},
        "encoder2": {"group_width": 36, "conv": {"start_frequency": 36, "end_frequency": 72, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 5, "stride": 3, "padding": 1}},
        "encoder3": {"group_width": 60, "conv": {"start_frequency": 72, "end_frequency": 132, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 10, "stride": 6, "padding": 2}},
        "encoder4": {"group_width": 80, "conv": {"start_frequency": 132, "end_frequency": 212, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 16, "stride": 8, "padding": 4}},
        "encoder5": {"group_width": 120, "conv": {"start_frequency": 212, "end_frequency": 332, "in_channels": 1,
                                                  "out_channels": 32, "kernel_size": 22, "stride": 12, "padding": 5}},
        "encoder6": {"group_width": 181, "conv": {"start_frequency": 332, "end_frequency": 513, "in_channels": 1,
                                                  "out_channels": 32, "kernel_size": 31, "stride": 18, "padding": 6}}
    }
    merge_split: dict = {"channels": 128, "bands": 32, "compress_rate": 2}
    bands_num_in_groups: Tuple[int] = get_sub_bands(sub_band_encoder)[0]
    band_width_in_groups: Tuple[int] = get_sub_bands(sub_band_encoder)[1]

    sub_band_decoder: Dict[str, dict] = {f"decoder{idx}": {"in_features": 64, "out_features": width}
                                         for idx, width in enumerate(band_width_in_groups)}

    dual_path_extension: dict = {
        "num_modules": 3,
        "parameters": {"input_size": 16, "intra_hidden_size": 16, "inter_hidden_size": 16,
                       "groups": 8, "rnn_type": "GRU"}
    }

    @field_validator("sub_band_decoder")
    def sub_band_decoder_validate(cls, decoders):
        for decoder in decoders:
            if decoder["out_feature"] < 2:
                raise ValueError(f"values should > 2, but got {decoder['out_feature']}")
            
class TrainConfig48kHzEnc2x(BaseModel):
    sample_rate: int = 48_000
    n_fft: int = 1024
    hop_length: int = 512
    train_frames: int = 62
    train_points: int = (train_frames - 1) * hop_length

    full_band_encoder: Dict[str, dict] = {
        "encoder1": {"in_channels": 2, "out_channels": 4, "kernel_size": 6, "stride": 2, "padding": 2},
        "encoder2": {"in_channels": 4, "out_channels": 16, "kernel_size": 8, "stride": 2, "padding": 3},
        "encoder3": {"in_channels": 16, "out_channels": 32, "kernel_size": 6, "stride": 2, "padding": 2},
        # "encoder4": {"in_channels": 16, "out_channels": 32, "kernel_size": 8, "stride": 2, "padding": 2}
    }
    full_band_decoder: Dict[str, dict] = {
        "decoder1": {"in_channels": 64, "out_channels": 16, "kernel_size": 6, "stride": 2, "padding": 2},
        "decoder2": {"in_channels": 32, "out_channels": 4, "kernel_size": 8, "stride": 2, "padding": 3},
        "decoder3": {"in_channels": 8, "out_channels": 2, "kernel_size": 6, "stride": 2, "padding": 2}
    }

    sub_band_encoder: Dict[str, dict] = {
        "encoder1": {"group_width": 40, "conv": {"start_frequency": 0, "end_frequency": 40, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 12, "stride": 4, "padding": 0}}, # 8
        "encoder2": {"group_width": 40, "conv": {"start_frequency": 40, "end_frequency": 80, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 12, "stride": 4, "padding": 0}}, # 8
        "encoder3": {"group_width": 42, "conv": {"start_frequency": 80, "end_frequency": 122, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 12, "stride": 6, "padding": 0}}, # 6
        "encoder4": {"group_width": 42, "conv": {"start_frequency": 122, "end_frequency": 164, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 12, "stride": 6, "padding": 0}}, # 6
        "encoder5": {"group_width": 48, "conv": {"start_frequency": 164, "end_frequency": 212, "in_channels": 1,
                                                  "out_channels": 32, "kernel_size": 13, "stride": 7, "padding": 0}}, # 6
        "encoder6": {"group_width": 48, "conv": {"start_frequency": 212, "end_frequency": 260, "in_channels": 1,
                                                  "out_channels": 32, "kernel_size": 13, "stride": 7, "padding": 0}}, # 6
        "encoder7": {"group_width": 48, "conv": {"start_frequency": 260, "end_frequency": 308, "in_channels": 1,
                                                  "out_channels": 32, "kernel_size": 13, "stride": 7, "padding": 0}}, # 6
        "encoder8": {"group_width": 60, "conv": {"start_frequency": 308, "end_frequency": 368, "in_channels": 1,
                                                  "out_channels": 32, "kernel_size": 15, "stride": 9, "padding": 0}}, # 6
        "encoder9": {"group_width": 60, "conv": {"start_frequency": 368, "end_frequency": 428, "in_channels": 1,
                                                  "out_channels": 32, "kernel_size": 15, "stride": 9, "padding": 0}}, # 6
        "encoder10": {"group_width": 85, "conv": {"start_frequency": 428, "end_frequency": 513, "in_channels": 1,
                                                  "out_channels": 32, "kernel_size": 20, "stride": 13, "padding": 0}}, # 6
    }
    merge_split: dict = {"channels": 128, "bands": 32, "compress_rate": 2}
    bands_num_in_groups: Tuple[int] = get_sub_bands(sub_band_encoder)[0]
    band_width_in_groups: Tuple[int] = get_sub_bands(sub_band_encoder)[1]

    sub_band_decoder: Dict[str, dict] = {f"decoder{idx}": {"in_features": 64, "out_features": width}
                                         for idx, width in enumerate(band_width_in_groups)}

    dual_path_extension: dict = {
        "num_modules": 3,
        "parameters": {"input_size": 16, "intra_hidden_size": 16, "inter_hidden_size": 16,
                       "groups": 8, "rnn_type": "GRU"}
    }

    @field_validator("sub_band_decoder")
    def sub_band_decoder_validate(cls, decoders):
        for decoder in decoders:
            if decoder["out_feature"] < 2:
                raise ValueError(f"values should > 2, but got {decoder['out_feature']}")
            
class TrainConfig48kHzEnc2x_ver2(BaseModel):
    sample_rate: int = 48_000
    n_fft: int = 1024
    hop_length: int = 512
    train_frames: int = 62
    train_points: int = (train_frames - 1) * hop_length

    full_band_encoder: Dict[str, dict] = {
        "encoder1": {"in_channels": 2, "out_channels": 4, "kernel_size": 6, "stride": 2, "padding": 2},
        "encoder2": {"in_channels": 4, "out_channels": 16, "kernel_size": 8, "stride": 2, "padding": 3},
        "encoder3": {"in_channels": 16, "out_channels": 32, "kernel_size": 6, "stride": 2, "padding": 2},
        # "encoder4": {"in_channels": 16, "out_channels": 32, "kernel_size": 8, "stride": 2, "padding": 2}
    }
    full_band_decoder: Dict[str, dict] = {
        "decoder1": {"in_channels": 64, "out_channels": 16, "kernel_size": 6, "stride": 2, "padding": 2},
        "decoder2": {"in_channels": 32, "out_channels": 4, "kernel_size": 8, "stride": 2, "padding": 3},
        "decoder3": {"in_channels": 8, "out_channels": 2, "kernel_size": 6, "stride": 2, "padding": 2}
    }

    sub_band_encoder: Dict[str, dict] = {
        "encoder1": {"group_width": 18, "conv": {"start_frequency": 0, "end_frequency": 18, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 5, "stride": 3, "padding": 1}}, # 8
        "encoder2": {"group_width": 18, "conv": {"start_frequency": 18, "end_frequency": 36, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 5, "stride": 3, "padding": 1}}, # 8
        "encoder3": {"group_width": 48, "conv": {"start_frequency": 36, "end_frequency": 84, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 13, "stride": 6, "padding": 0}}, # 6
        "encoder4": {"group_width": 48, "conv": {"start_frequency": 84, "end_frequency": 132, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 13, "stride": 6, "padding": 0}}, # 6
        "encoder5": {"group_width": 66, "conv": {"start_frequency": 132, "end_frequency": 198, "in_channels": 1,
                                                  "out_channels": 32, "kernel_size": 16, "stride": 9, "padding": 0}}, # 6
        "encoder6": {"group_width": 66, "conv": {"start_frequency": 198, "end_frequency": 264, "in_channels": 1,
                                                  "out_channels": 32, "kernel_size": 16, "stride": 9, "padding": 0}}, # 6
        "encoder7": {"group_width": 120, "conv": {"start_frequency": 264, "end_frequency": 384, "in_channels": 1,
                                                  "out_channels": 32, "kernel_size": 16, "stride": 9, "padding": 0}}, # 6
        "encoder8": {"group_width": 129, "conv": {"start_frequency": 384, "end_frequency": 513, "in_channels": 1,
                                                  "out_channels": 32, "kernel_size": 20, "stride": 7, "padding": 0}}, # 6
    }
    merge_split: dict = {"channels": 128, "bands": 32, "compress_rate": 2}
    bands_num_in_groups: Tuple[int] = get_sub_bands(sub_band_encoder)[0]
    band_width_in_groups: Tuple[int] = get_sub_bands(sub_band_encoder)[1]

    sub_band_decoder: Dict[str, dict] = {f"decoder{idx}": {"in_features": 64, "out_features": width}
                                         for idx, width in enumerate(band_width_in_groups)}

    dual_path_extension: dict = {
        "num_modules": 3,
        "parameters": {"input_size": 16, "intra_hidden_size": 16, "inter_hidden_size": 16,
                       "groups": 8, "rnn_type": "GRU"}
    }

    @field_validator("sub_band_decoder")
    def sub_band_decoder_validate(cls, decoders):
        for decoder in decoders:
            if decoder["out_feature"] < 2:
                raise ValueError(f"values should > 2, but got {decoder['out_feature']}")


class TrainConfig48kHzEnc2x_ver3(BaseModel):
    sample_rate: int = 48_000
    n_fft: int = 1024
    hop_length: int = 512
    train_frames: int = 62
    train_points: int = (train_frames - 1) * hop_length

    full_band_encoder: Dict[str, dict] = {
        "encoder1": {"in_channels": 2, "out_channels": 4, "kernel_size": 6, "stride": 2, "padding": 2},
        "encoder2": {"in_channels": 4, "out_channels": 16, "kernel_size": 8, "stride": 2, "padding": 3},
        "encoder3": {"in_channels": 16, "out_channels": 32, "kernel_size": 6, "stride": 2, "padding": 2},
        # "encoder4": {"in_channels": 16, "out_channels": 32, "kernel_size": 8, "stride": 2, "padding": 2}
    }
    full_band_decoder: Dict[str, dict] = {
        "decoder1": {"in_channels": 64, "out_channels": 16, "kernel_size": 6, "stride": 2, "padding": 2},
        "decoder2": {"in_channels": 32, "out_channels": 4, "kernel_size": 8, "stride": 2, "padding": 3},
        "decoder3": {"in_channels": 8, "out_channels": 2, "kernel_size": 6, "stride": 2, "padding": 2}
    }

    sub_band_encoder: Dict[str, dict] = {
        "encoder1": {"group_width": 18, "conv": {"start_frequency": 0, "end_frequency": 18, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 5, "stride": 3, "padding": 1}}, # 8
        "encoder2": {"group_width": 18, "conv": {"start_frequency": 18, "end_frequency": 36, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 5, "stride": 3, "padding": 1}}, # 8
        "encoder3": {"group_width": 48, "conv": {"start_frequency": 36, "end_frequency": 84, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 13, "stride": 6, "padding": 0}}, # 6
        "encoder4": {"group_width": 48, "conv": {"start_frequency": 84, "end_frequency": 132, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 13, "stride": 6, "padding": 0}}, # 6
        "encoder5": {"group_width": 66, "conv": {"start_frequency": 132, "end_frequency": 198, "in_channels": 1,
                                                  "out_channels": 32, "kernel_size": 16, "stride": 9, "padding": 0}}, # 6
        "encoder6": {"group_width": 66, "conv": {"start_frequency": 198, "end_frequency": 264, "in_channels": 1,
                                                  "out_channels": 32, "kernel_size": 16, "stride": 9, "padding": 0}}, # 6
        "encoder7": {"group_width": 120, "conv": {"start_frequency": 264, "end_frequency": 384, "in_channels": 1,
                                                  "out_channels": 32, "kernel_size": 16, "stride": 9, "padding": 0}}, # 6
        "encoder8": {"group_width": 129, "conv": {"start_frequency": 384, "end_frequency": 513, "in_channels": 1,
                                                  "out_channels": 32, "kernel_size": 20, "stride": 7, "padding": 0}}, # 6
    }
    merge_split: dict = {"channels": 128, "bands": 32, "compress_rate": 2}
    bands_num_in_groups: Tuple[int] = get_sub_bands(sub_band_encoder)[0]
    band_width_in_groups: Tuple[int] = get_sub_bands(sub_band_encoder)[1]

    sub_band_decoder: Dict[str, dict] = {f"decoder{idx}": {"in_features": 64, "out_features": width}
                                         for idx, width in enumerate(band_width_in_groups)}

    dual_path_extension: dict = {
        "num_modules": 3,
        "parameters": {"input_size": 16, "intra_hidden_size": 64, "inter_hidden_size": 64,
                       "groups": 8, "rnn_type": "GRU"}
    }

    @field_validator("sub_band_decoder")
    def sub_band_decoder_validate(cls, decoders):
        for decoder in decoders:
            if decoder["out_feature"] < 2:
                raise ValueError(f"values should > 2, but got {decoder['out_feature']}")


class TrainConfig48kHzEnc2x_ver1(BaseModel):
    sample_rate: int = 48_000
    n_fft: int = 1024
    hop_length: int = 512
    train_frames: int = 62
    train_points: int = (train_frames - 1) * hop_length

    full_band_encoder: Dict[str, dict] = {
        "encoder1": {"in_channels": 2, "out_channels": 4, "kernel_size": 4, "stride": 1, "padding": 0},
        "encoder2": {"in_channels": 4, "out_channels": 8, "kernel_size": 6, "stride": 2, "padding": 1},
        "encoder3": {"in_channels": 8, "out_channels": 16, "kernel_size": 6, "stride": 2, "padding": 1},
        "encoder4": {"in_channels": 16, "out_channels": 32, "kernel_size": 6, "stride": 2, "padding": 3}
    }
    full_band_decoder: Dict[str, dict] = {
        "decoder1": {"in_channels": 64, "out_channels": 16, "kernel_size": 6, "stride": 2, "padding": 3, "output_padding": 0},
        "decoder2": {"in_channels": 32, "out_channels": 8, "kernel_size": 6, "stride": 2, "padding": 1, "output_padding": 0},
        "decoder3": {"in_channels": 16, "out_channels": 4, "kernel_size": 6, "stride": 2, "padding": 1, "output_padding": 0},
        "decoder4": {"in_channels": 8, "out_channels": 2, "kernel_size": 4, "stride": 1, "padding": 0, "output_padding": 0}
    }

    sub_band_encoder: Dict[str, dict] = {
        "encoder1": {"group_width": 40, "conv": {"start_frequency": 0, "end_frequency": 40, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 12, "stride": 4, "padding": 0}}, # 8
        "encoder2": {"group_width": 40, "conv": {"start_frequency": 40, "end_frequency": 80, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 12, "stride": 4, "padding": 0}}, # 8
        "encoder3": {"group_width": 42, "conv": {"start_frequency": 80, "end_frequency": 122, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 12, "stride": 6, "padding": 0}}, # 6
        "encoder4": {"group_width": 42, "conv": {"start_frequency": 122, "end_frequency": 164, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 12, "stride": 6, "padding": 0}}, # 6
        "encoder5": {"group_width": 48, "conv": {"start_frequency": 164, "end_frequency": 212, "in_channels": 1,
                                                  "out_channels": 32, "kernel_size": 13, "stride": 7, "padding": 0}}, # 6
        "encoder6": {"group_width": 48, "conv": {"start_frequency": 212, "end_frequency": 260, "in_channels": 1,
                                                  "out_channels": 32, "kernel_size": 13, "stride": 7, "padding": 0}}, # 6
        "encoder7": {"group_width": 48, "conv": {"start_frequency": 260, "end_frequency": 308, "in_channels": 1,
                                                  "out_channels": 32, "kernel_size": 13, "stride": 7, "padding": 0}}, # 6
        "encoder8": {"group_width": 60, "conv": {"start_frequency": 308, "end_frequency": 368, "in_channels": 1,
                                                  "out_channels": 32, "kernel_size": 15, "stride": 9, "padding": 0}}, # 6
        "encoder9": {"group_width": 60, "conv": {"start_frequency": 368, "end_frequency": 428, "in_channels": 1,
                                                  "out_channels": 32, "kernel_size": 15, "stride": 9, "padding": 0}}, # 6
        "encoder10": {"group_width": 85, "conv": {"start_frequency": 428, "end_frequency": 513, "in_channels": 1,
                                                  "out_channels": 32, "kernel_size": 20, "stride": 13, "padding": 0}}, # 6
    }
    merge_split: dict = {"channels": 128, "bands": 32, "compress_rate": 2}
    bands_num_in_groups: Tuple[int] = get_sub_bands(sub_band_encoder)[0]
    band_width_in_groups: Tuple[int] = get_sub_bands(sub_band_encoder)[1]

    sub_band_decoder: Dict[str, dict] = {f"decoder{idx}": {"in_features": 64, "out_features": width}
                                         for idx, width in enumerate(band_width_in_groups)}

    dual_path_extension: dict = {
        "num_modules": 3,
        "parameters": {"input_size": 16, "intra_hidden_size": 16, "inter_hidden_size": 16,
                       "groups": 8, "rnn_type": "GRU"}
    }

    @field_validator("sub_band_decoder")
    def sub_band_decoder_validate(cls, decoders):
        for decoder in decoders:
            if decoder["out_feature"] < 2:
                raise ValueError(f"values should > 2, but got {decoder['out_feature']}")

class TrainConfigLarge(BaseModel):
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 256
    train_frames: int = 62
    train_points: int = (train_frames - 1) * hop_length

    full_band_encoder: Dict[str, dict] = {
        "encoder1": {"conv": Conv1d, "in_channels": 2, "out_channels": 4, "kernel_size": 6, "stride": 2, "padding": 2},
        "encoder2": {"conv": Conv1d, "in_channels": 4, "out_channels": 16, "kernel_size": 8, "stride": 2, "padding": 3},
        "encoder3": {"conv": Conv1d, "in_channels": 16, "out_channels": 32 * 2, "kernel_size": 6, "stride": 2, "padding": 2}
    }
    full_band_decoder: Dict[str, dict] = {
        "decoder1": {"conv": Conv1d, "conv_transposed": ConvTranspose1d,
                     "in_channels": 64 * 2, "out_channels": 16, "kernel_size": 6, "stride": 2, "padding": 2},
        "decoder2": {"conv": Conv1d, "conv_transposed": ConvTranspose1d,
                     "in_channels": 32, "out_channels": 4, "kernel_size": 8, "stride": 2, "padding": 3},
        "decoder3": {"conv": Conv1d, "conv_transposed": ConvTranspose1d,
                     "in_channels": 8, "out_channels": 2, "kernel_size": 6, "stride": 2, "padding": 2}
    }

    sub_band_encoder: Dict[str, dict] = {
        "encoder1": {"group_width": 16, "conv": {"conv": Conv1d, "start_frequency": 0, "end_frequency": 16, "in_channels": 1,
                                                 "out_channels": 32 * 2, "kernel_size": 4, "stride": 2, "padding": 1}},
        "encoder2": {"group_width": 18, "conv": {"conv": Conv1d, "start_frequency": 16, "end_frequency": 34, "in_channels": 1,
                                                 "out_channels": 32 * 2, "kernel_size": 7, "stride": 3, "padding": 2}},
        "encoder3": {"group_width": 36, "conv": {"conv": Conv1d, "start_frequency": 34, "end_frequency": 70, "in_channels": 1,
                                                 "out_channels": 32 * 2, "kernel_size": 11, "stride": 5, "padding": 2}},
        "encoder4": {"group_width": 66, "conv": {"conv": Conv1d, "start_frequency": 70, "end_frequency": 136, "in_channels": 1,
                                                 "out_channels": 32 * 2, "kernel_size": 20, "stride": 10, "padding": 4}},
        "encoder5": {"group_width": 121, "conv": {"conv": Conv1d, "start_frequency": 136, "end_frequency": 257, "in_channels": 1,
                                                  "out_channels": 32 * 2, "kernel_size": 30, "stride": 20, "padding": 5}}
    }
    merge_split: dict = {"conv": Conv1d, "channels": 64, "bands": 32 * 2, "compress_rate": 2}
    bands_num_in_groups: Tuple[int] = get_sub_bands(sub_band_encoder)[0]
    band_width_in_groups: Tuple[int] = get_sub_bands(sub_band_encoder)[1]

    sub_band_decoder: Dict[str, dict] = {f"decoder{idx}": {"in_features": 64 * 2, "out_features": width}
                                         for idx, width in enumerate(band_width_in_groups)}

    dual_path_extension: dict = {
        "num_modules": 2,
        "parameters": {"input_size": 16 * 2, "intra_hidden_size": 16, "inter_hidden_size": 16,
                       "groups": 8, "rnn_type": "GRU"}
    }

    @field_validator("sub_band_decoder")
    def sub_band_decoder_validate(cls, decoders):
        for decoder in decoders:
            if decoder["out_feature"] < 2:
                raise ValueError(f"values should > 2, but got {decoder['out_feature']}")
            
class TrainConfigLarge1(BaseModel):
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 256
    train_frames: int = 62
    train_points: int = (train_frames - 1) * hop_length

    full_band_encoder: Dict[str, dict] = {
        "encoder1": {"conv": CausalConv1d, "in_channels": 2, "out_channels": 4, "kernel_size": 6, "stride": 2, "padding": 2},
        "encoder2": {"conv": CausalConv1d, "in_channels": 4, "out_channels": 16, "kernel_size": 8, "stride": 2, "padding": 3},
        "encoder3": {"conv": CausalConv1d, "in_channels": 16, "out_channels": 32 * 2, "kernel_size": 6, "stride": 2, "padding": 2}
    }
    full_band_decoder: Dict[str, dict] = {
        "decoder1": {"conv": CausalConv1d, "conv_transposed": CausalConvTranspose1d,
                     "in_channels": 64 * 2, "out_channels": 16, "kernel_size": 6, "stride": 2, "padding": 2},
        "decoder2": {"conv": CausalConv1d, "conv_transposed": CausalConvTranspose1d,
                     "in_channels": 32, "out_channels": 4, "kernel_size": 8, "stride": 2, "padding": 3},
        "decoder3": {"conv": CausalConv1d, "conv_transposed": CausalConvTranspose1d,
                     "in_channels": 8, "out_channels": 2, "kernel_size": 6, "stride": 2, "padding": 2}
    }

    sub_band_encoder: Dict[str, dict] = {
        "encoder1": {"group_width": 18, "conv": {"conv": CausalConv1d, "start_frequency": 0, "end_frequency": 18, "in_channels": 1,
                                                 "out_channels": 32 * 2, "kernel_size": 4, "stride": 2, "padding": 0}},
        "encoder2": {"group_width": 47, "conv": {"conv": CausalConv1d, "start_frequency": 18, "end_frequency": 65, "in_channels": 1,
                                                 "out_channels": 32 * 2, "kernel_size": 8, "stride": 5, "padding": 0}},
        "encoder3": {"group_width": 67, "conv": {"conv": CausalConv1d, "start_frequency": 65, "end_frequency": 132, "in_channels": 1,
                                                 "out_channels": 32 * 2, "kernel_size": 12, "stride": 7, "padding": 0}},
        "encoder4": {"group_width": 125, "conv": {"conv": CausalConv1d, "start_frequency": 132, "end_frequency": 257, "in_channels": 1,
                                                 "out_channels": 32 * 2, "kernel_size": 20, "stride": 14, "padding": 0}},
    }

    # sub_band_encoder: Dict[str, dict] = {
    #     "encoder1": {"group_width": 12, "conv": {"conv": CausalConv1d, "start_frequency": 0, "end_frequency": 12, "in_channels": 1, # group_width % out == 0
    #                                              "out_channels": 32 * 2, "kernel_size": 4, "stride": 2, "padding": 1}},
    #     "encoder2": {"group_width": 16, "conv": {"conv": CausalConv1d, "start_frequency": 12, "end_frequency": 28, "in_channels": 1,
    #                                              "out_channels": 32 * 2, "kernel_size": 6, "stride": 3, "padding": 2}},
    #     "encoder3": {"group_width": 20, "conv": {"conv": CausalConv1d, "start_frequency": 28, "end_frequency": 48, "in_channels": 1,
    #                                              "out_channels": 32 * 2, "kernel_size": 8, "stride": 4, "padding": 0}},
    #     "encoder4": {"group_width": 28, "conv": {"conv": CausalConv1d, "start_frequency": 48, "end_frequency": 76, "in_channels": 1,
    #                                              "out_channels": 32 * 2, "kernel_size": 10, "stride": 5, "padding": 0}},
    #     "encoder5": {"group_width": 32, "conv": {"conv": CausalConv1d, "start_frequency": 76, "end_frequency": 108, "in_channels": 1,
    #                                              "out_channels": 32 * 2, "kernel_size": 12, "stride": 6, "padding": 0}},    
    #     "encoder6": {"group_width": 36, "conv": {"conv": CausalConv1d, "start_frequency": 108, "end_frequency": 144, "in_channels": 1,
    #                                              "out_channels": 32 * 2, "kernel_size": 14, "stride": 7, "padding": 0}},
    #     "encoder7": {"group_width": 113, "conv": {"conv": CausalConv1d, "start_frequency": 144, "end_frequency": 257, "in_channels": 1,
    #                                              "out_channels": 32 * 2, "kernel_size": 20, "stride": 13, "padding": 0}},                                      
    # }

    merge_split: dict = {"conv": CausalConv1d, "channels": 64, "bands": 32 * 2, "compress_rate": 2}
    bands_num_in_groups: Tuple[int] = get_sub_bands(sub_band_encoder)[0]
    band_width_in_groups: Tuple[int] = [x["group_width"] for x in sub_band_encoder.values()] # get_sub_bands(sub_band_encoder)[1]
    band_width_in_groups[-1] -= 1

    sub_band_decoder: Dict[str, dict] = {f"decoder{idx}": {"in_features": 64 * 2 * num, "out_features": width}
                                         for idx, (num, width) in enumerate(zip(bands_num_in_groups, band_width_in_groups))}

    dual_path_extension: dict = {
        "num_modules": 2,
        "parameters": {"input_size": 16 * 2, "intra_hidden_size": 16, "inter_hidden_size": 16,
                       "groups": 8, "rnn_type": "GRU"}
    }

    @field_validator("sub_band_decoder")
    def sub_band_decoder_validate(cls, decoders):
        for decoder in decoders:
            if decoder["out_feature"] < 2:
                raise ValueError(f"values should > 2, but got {decoder['out_feature']}")

class TrainConfigRNN(BaseModel):
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 256
    train_frames: int = 62
    train_points: int = (train_frames - 1) * hop_length

    full_band_encoder: Dict[str, dict] = {
        "encoder1": {"in_channels": 2, "out_channels": 4, "kernel_size": 6, "stride": 2, "padding": 2},
        "encoder2": {"in_channels": 4, "out_channels": 16, "kernel_size": 8, "stride": 2, "padding": 3},
        "encoder3": {"in_channels": 16, "out_channels": 32, "kernel_size": 6, "stride": 2, "padding": 2}
    }
    full_band_decoder: Dict[str, dict] = {
        "decoder1": {"in_channels": 64, "out_channels": 16, "kernel_size": 6, "stride": 2, "padding": 2},
        "decoder2": {"in_channels": 32, "out_channels": 4, "kernel_size": 8, "stride": 2, "padding": 3},
        "decoder3": {"in_channels": 8, "out_channels": 2, "kernel_size": 6, "stride": 2, "padding": 2}
    }

    sub_band_encoder: Dict[str, dict] = {
        "encoder1": {"group_width": 16, "conv": {"start_frequency": 0, "end_frequency": 16, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 4, "stride": 2, "padding": 1}},
        "encoder2": {"group_width": 18, "conv": {"start_frequency": 16, "end_frequency": 34, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 7, "stride": 3, "padding": 2}},
        "encoder3": {"group_width": 36, "conv": {"start_frequency": 34, "end_frequency": 70, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 11, "stride": 5, "padding": 2}},
        "encoder4": {"group_width": 66, "conv": {"start_frequency": 70, "end_frequency": 136, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 20, "stride": 10, "padding": 4}},
        "encoder5": {"group_width": 121, "conv": {"start_frequency": 136, "end_frequency": 257, "in_channels": 1,
                                                  "out_channels": 32, "kernel_size": 30, "stride": 20, "padding": 5}}
    }
    merge_split: dict = {"channels": 64, "bands": 32, "compress_rate": 2}
    bands_num_in_groups: Tuple[int] = get_sub_bands(sub_band_encoder)[0]
    band_width_in_groups: Tuple[int] = get_sub_bands(sub_band_encoder)[1]

    sub_band_decoder: Dict[str, dict] = {f"decoder{idx}": {"in_features": 64, "out_features": width}
                                         for idx, width in enumerate(band_width_in_groups)}

    dual_path_extension: dict = {
        "num_modules": 3,
        "parameters": {"input_size": 16, "intra_hidden_size": 8, "inter_hidden_size": 8,
                       "groups": 8, "rnn_type": "GRU"}
    }

    @field_validator("sub_band_decoder")
    def sub_band_decoder_validate(cls, decoders):
        for decoder in decoders:
            if decoder["out_feature"] < 2:
                raise ValueError(f"values should > 2, but got {decoder['out_feature']}")

class TrainConfigRNN1(BaseModel):
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 256
    train_frames: int = 62
    train_points: int = (train_frames - 1) * hop_length

    full_band_encoder: Dict[str, dict] = {
        "encoder1": {"in_channels": 2, "out_channels": 4, "kernel_size": 6, "stride": 2, "padding": 2},
        "encoder2": {"in_channels": 4, "out_channels": 16, "kernel_size": 8, "stride": 2, "padding": 3},
        "encoder3": {"in_channels": 16, "out_channels": 32, "kernel_size": 6, "stride": 2, "padding": 2}
    }
    full_band_decoder: Dict[str, dict] = {
        "decoder1": {"in_channels": 64, "out_channels": 16, "kernel_size": 6, "stride": 2, "padding": 2},
        "decoder2": {"in_channels": 32, "out_channels": 4, "kernel_size": 8, "stride": 2, "padding": 3},
        "decoder3": {"in_channels": 8, "out_channels": 2, "kernel_size": 6, "stride": 2, "padding": 2}
    }

    sub_band_encoder: Dict[str, dict] = {
        "encoder1": {"group_width": 16, "conv": {"start_frequency": 0, "end_frequency": 16, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 4, "stride": 2, "padding": 1}},
        "encoder2": {"group_width": 18, "conv": {"start_frequency": 16, "end_frequency": 34, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 7, "stride": 3, "padding": 2}},
        "encoder3": {"group_width": 36, "conv": {"start_frequency": 34, "end_frequency": 70, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 11, "stride": 5, "padding": 2}},
        "encoder4": {"group_width": 66, "conv": {"start_frequency": 70, "end_frequency": 136, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 20, "stride": 10, "padding": 4}},
        "encoder5": {"group_width": 121, "conv": {"start_frequency": 136, "end_frequency": 257, "in_channels": 1,
                                                  "out_channels": 32, "kernel_size": 30, "stride": 20, "padding": 5}}
    }
    merge_split: dict = {"channels": 64, "bands": 32, "compress_rate": 2}
    bands_num_in_groups: Tuple[int] = get_sub_bands(sub_band_encoder)[0]
    band_width_in_groups: Tuple[int] = get_sub_bands(sub_band_encoder)[1]

    sub_band_decoder: Dict[str, dict] = {f"decoder{idx}": {"in_features": 64, "out_features": width}
                                         for idx, width in enumerate(band_width_in_groups)}

    dual_path_extension: dict = {
        "num_modules": 2,
        "parameters": {"input_size": 16, "intra_hidden_size": 16, "inter_hidden_size": 16,
                       "groups": 8, "rnn_type": "GRU"}
    }

    @field_validator("sub_band_decoder")
    def sub_band_decoder_validate(cls, decoders):
        for decoder in decoders:
            if decoder["out_feature"] < 2:
                raise ValueError(f"values should > 2, but got {decoder['out_feature']}")

class TrainConfigRNN2(BaseModel):
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 256
    train_frames: int = 62
    train_points: int = (train_frames - 1) * hop_length

    full_band_encoder: Dict[str, dict] = {
        "encoder1": {"in_channels": 2, "out_channels": 4, "kernel_size": 6, "stride": 2, "padding": 2},
        "encoder2": {"in_channels": 4, "out_channels": 16, "kernel_size": 8, "stride": 2, "padding": 3},
        "encoder3": {"in_channels": 16, "out_channels": 32, "kernel_size": 6, "stride": 2, "padding": 2}
    }
    full_band_decoder: Dict[str, dict] = {
        "decoder1": {"in_channels": 64, "out_channels": 16, "kernel_size": 6, "stride": 2, "padding": 2},
        "decoder2": {"in_channels": 32, "out_channels": 4, "kernel_size": 8, "stride": 2, "padding": 3},
        "decoder3": {"in_channels": 8, "out_channels": 2, "kernel_size": 6, "stride": 2, "padding": 2}
    }

    sub_band_encoder: Dict[str, dict] = {
        "encoder1": {"group_width": 16, "conv": {"start_frequency": 0, "end_frequency": 16, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 4, "stride": 2, "padding": 1}},
        "encoder2": {"group_width": 18, "conv": {"start_frequency": 16, "end_frequency": 34, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 7, "stride": 3, "padding": 2}},
        "encoder3": {"group_width": 36, "conv": {"start_frequency": 34, "end_frequency": 70, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 11, "stride": 5, "padding": 2}},
        "encoder4": {"group_width": 66, "conv": {"start_frequency": 70, "end_frequency": 136, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 20, "stride": 10, "padding": 4}},
        "encoder5": {"group_width": 121, "conv": {"start_frequency": 136, "end_frequency": 257, "in_channels": 1,
                                                  "out_channels": 32, "kernel_size": 30, "stride": 20, "padding": 5}}
    }
    merge_split: dict = {"channels": 64, "bands": 32, "compress_rate": 4}
    bands_num_in_groups: Tuple[int] = get_sub_bands(sub_band_encoder)[0]
    band_width_in_groups: Tuple[int] = get_sub_bands(sub_band_encoder)[1]

    sub_band_decoder: Dict[str, dict] = {f"decoder{idx}": {"in_features": 64, "out_features": width}
                                         for idx, width in enumerate(band_width_in_groups)}

    dual_path_extension: dict = {
        "num_modules": 3,
        "parameters": {"input_size": 8, "intra_hidden_size": 4, "inter_hidden_size": 4,
                       "groups": 4, "rnn_type": "GRU"}
    }

    @field_validator("sub_band_decoder")
    def sub_band_decoder_validate(cls, decoders):
        for decoder in decoders:
            if decoder["out_feature"] < 2:
                raise ValueError(f"values should > 2, but got {decoder['out_feature']}")

if __name__ == "__main__":
    test_configs = TrainConfig()

    for (decoder_name, parameters), _, in zip(test_configs.sub_band_encoder.items(), test_configs.bands_num_in_groups):
        print(parameters)
