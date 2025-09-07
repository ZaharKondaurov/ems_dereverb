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

class TrainConfigLarge(BaseModel):
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
        "encoder1": {"group_width": 16, "conv": {"conv": CausalConv1d, "start_frequency": 0, "end_frequency": 16, "in_channels": 1,
                                                 "out_channels": 32 * 2, "kernel_size": 4, "stride": 2, "padding": 1}},
        "encoder2": {"group_width": 18, "conv": {"conv": CausalConv1d, "start_frequency": 16, "end_frequency": 34, "in_channels": 1,
                                                 "out_channels": 32 * 2, "kernel_size": 7, "stride": 3, "padding": 2}},
        "encoder3": {"group_width": 36, "conv": {"conv": CausalConv1d, "start_frequency": 34, "end_frequency": 70, "in_channels": 1,
                                                 "out_channels": 32 * 2, "kernel_size": 11, "stride": 5, "padding": 2}},
        "encoder4": {"group_width": 66, "conv": {"conv": CausalConv1d, "start_frequency": 70, "end_frequency": 136, "in_channels": 1,
                                                 "out_channels": 32 * 2, "kernel_size": 20, "stride": 10, "padding": 4}},
        "encoder5": {"group_width": 121, "conv": {"conv": CausalConv1d, "start_frequency": 136, "end_frequency": 257, "in_channels": 1,
                                                  "out_channels": 32 * 2, "kernel_size": 30, "stride": 20, "padding": 5}}
    }
    merge_split: dict = {"conv": CausalConv1d, "channels": 64, "bands": 32 * 2, "compress_rate": 2}
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
