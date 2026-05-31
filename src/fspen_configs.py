from typing import Dict, Tuple, List, Union

from pydantic import BaseModel, ConfigDict

import torch.nn as nn


class FSPENBaseConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


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

def get_cnn(in_channels, kernel_size, padding, stride, out_channels, layers, channel_step=2):
    encoder: Dict[str, dict] = {}

    step = channel_step
    if isinstance(channel_step, tuple):
        step = channel_step[0]

    for i in range(1, layers + 1):
        encoder[f"encoder{i}"] = {"in_channels": in_channels, "out_channels": min(in_channels * step, out_channels), "kernel_size": kernel_size, "stride": stride, "padding": padding}
        if i == layers:
            encoder[f"encoder{i}"] = {"in_channels": in_channels, "out_channels": out_channels, "kernel_size": kernel_size, "stride": stride, "padding": padding}
        in_channels = min(step * in_channels, out_channels)
        if isinstance(channel_step, tuple):
            step = channel_step[i % len(channel_step)]
        
        if i <= layers // 2:
            kernel_size += 2
            padding += 1
        else:
            kernel_size -= 2
            padding -= 1
        
    return encoder

def build_sub_band_decoder_params(sbe: Dict[str, dict], width: Tuple[int], overlap=None):
    params = list(sbe.values())
    sbd = {}
    start_idx = 0
    for i in range(len(params)):
        end_idx = start_idx + width[i]

        d = {"start_idx": start_idx, "end_idx": end_idx, "width": width[i], "convs": []}
        convs = list(params[i]["convs"])[::-1]
        for conv in convs:
            d["convs"].append({"in_channels": conv["out_channels"] * 2, "out_channels": conv["in_channels"], 
                               "kernel_size": conv["kernel_size"], "stride": conv["stride"], "padding": conv["padding"]})
        sbd[f"decoder{i + 1}"] = d
    return sbd

def get_widths(n_fft, num_sub_bands, overlap=None):
    widths = [n_fft // 2]
    for i in range(num_sub_bands - 1):
        widths.append(widths[-1] // 2)
        widths[-2] = widths[-2] // 2

    widths[0] += 1
    widths = widths[::-1]

    return widths

def get_full_band_decoder(full_band_encoder):
    full_band_decoder: Dict[str, dict] = {}
    for i, vals in enumerate(list(full_band_encoder.values())[::-1]):
        full_band_decoder[f"decoder{i}"] = {"in_channels": vals["out_channels"] * 2, "out_channels": vals["in_channels"], 
                                            "kernel_size": vals["kernel_size"], "stride": vals["stride"], "padding": vals["padding"]}
    return full_band_decoder

def get_sub_band_encoder(widths, sub_band_layers, channels, channel_step=2, overlap=None):
    start_freq = 0
    sub_band_encoder: Dict[str, dict] = {}
    for i in range(1, len(widths) + 1):
        sub_band_encoder[f"encoder{i}"] = {"group_width": widths[i - 1], 
                                           "bounds": {"start_frequency": start_freq, "end_frequency": start_freq + widths[i - 1]},
                                           "convs": []}
        sub_band_encoder[f"encoder{i}"]["convs"] = list(get_cnn(in_channels=1, kernel_size=4, stride=2, padding=1, out_channels=channels,
                                            layers=sub_band_layers, channel_step=channel_step).values())
        
        start_freq = start_freq + (widths[i - 1])
        
    return sub_band_encoder

def get_end_bands(widths, sub_band_layers):
    end_bands = []
    for width in widths:
        end_bands.append(width // (2 ** sub_band_layers))
    return end_bands
    

class TrainConfig(FSPENBaseConfig):
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 256
    train_frames: int = 62
    train_points: int = (train_frames - 1) * hop_length

    split_last: bool = False

    full_band_encoder: Dict[str, dict] = {
        "encoder1": {"in_channels": 2, "out_channels": 4, "kernel_size": 6, "stride": 2, "padding": 2,},
        "encoder2": {"in_channels": 4, "out_channels": 16, "kernel_size": 8, "stride": 2, "padding": 3,},
        "encoder3": {"in_channels": 16, "out_channels": 32, "kernel_size": 6, "stride": 2, "padding": 2,}
    }
    full_band_decoder: Dict[str, dict] = {
        "decoder1": {"in_channels": 64, "out_channels": 16, "kernel_size": 6, "stride": 2, "padding": 2,},
        "decoder2": {"in_channels": 32, "out_channels": 4, "kernel_size": 8, "stride": 2, "padding": 3,},
        "decoder3": {"in_channels": 8, "out_channels": 2, "kernel_size": 6, "stride": 2, "padding": 2,}
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
                       "groups": 8, "rnn_type": "GRU", "num_layers": 1}
    }

    num_bands_out: int = 32

    mag_act: Dict[str, nn.Module] = {"full": nn.ELU, "sub": nn.ReLU}
    last_act: Dict[str, nn.Module] = {"mag_act": nn.ReLU, "pha_act": nn.ELU}
    

class TrainConfig_baseline(FSPENBaseConfig):
    sample_rate: int = 48_000
    n_fft: int = 512
    hop_length: int = 256
    train_frames: int = 62
    train_points: int = (train_frames - 1) * hop_length

    split_last: bool = False

    full_band_encoder: Dict[str, dict] = {
        "encoder1": {"in_channels": 2, "out_channels": 4, "kernel_size": 6, "stride": 2, "padding": 2},
        "encoder2": {"in_channels": 4, "out_channels": 16, "kernel_size": 8, "stride": 2, "padding": 3},
        "encoder3": {"in_channels": 16, "out_channels": 32, "kernel_size": 6, "stride": 2, "padding": 2}
    }
    full_band_decoder: Dict[str, dict] = {
        "decoder1": {"in_channels": 64, "out_channels": 16, "kernel_size": 6, "stride": 2, "padding": 2, },
        "decoder2": {"in_channels": 32, "out_channels": 4, "kernel_size": 8, "stride": 2, "padding": 3,},
        "decoder3": {"in_channels": 8, "out_channels": 2, "kernel_size": 6, "stride": 2, "padding": 2,}
    }

    sub_band_encoder: Dict[str, dict] = {
        "encoder1": {"group_width": 16, "conv": {"start_frequency": 0, "end_frequency": 16, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 4, "stride": 2, "padding": 1,}},
        "encoder2": {"group_width": 18, "conv": {"start_frequency": 16, "end_frequency": 34, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 7, "stride": 3, "padding": 2,}},
        "encoder3": {"group_width": 36, "conv": {"start_frequency": 34, "end_frequency": 70, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 11, "stride": 5, "padding": 2,}},
        "encoder4": {"group_width": 66, "conv": {"start_frequency": 70, "end_frequency": 136, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 20, "stride": 10, "padding": 4,}},
        "encoder5": {"group_width": 121, "conv": {"start_frequency": 136, "end_frequency": 257, "in_channels": 1,
                                                  "out_channels": 32, "kernel_size": 30, "stride": 20, "padding": 5,}}
    }
    merge_split: dict = {"channels": 64, "bands": 32, "compress_rate": 2}
    bands_num_in_groups: Tuple[int] = get_sub_bands(sub_band_encoder)[0]
    band_width_in_groups: Tuple[int] = get_sub_bands(sub_band_encoder)[1]

    sub_band_decoder: Dict[str, dict] = {f"decoder{idx}": {"in_features": 64, "out_features": width}
                                         for idx, width in enumerate(band_width_in_groups)}

    dual_path_extension: dict = {
        "num_modules": 3,
        "parameters": {"input_size": 16, "intra_hidden_size": 16, "inter_hidden_size": 16,
                       "groups": 8, "rnn_type": "GRU", "num_layers": 1}
    }

    num_bands_out: int = 32

    mag_act: Dict[str, nn.Module] = {"full": nn.ELU, "sub": nn.ReLU}
    last_act: Dict[str, nn.Module] = {"mag_act": nn.ReLU, "pha_act": nn.ELU}
            

class TrainConfig_48khz(FSPENBaseConfig):
    sample_rate: int = 48000
    n_fft: int = 1024
    hop_length: int = 512
    train_frames: int = 62
    train_points: int = (train_frames - 1) * hop_length
    batch_size: int = 32

    split_last: bool = False

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
        "encoder1": {"group_width": 18, "conv": {"start_frequency": 0, "end_frequency": 18, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 5, "stride": 3, "padding": 1}}, # 6
        "encoder2": {"group_width": 18, "conv": {"start_frequency": 18, "end_frequency": 36, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 5, "stride": 3, "padding": 1}}, # 6
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

    num_bands_out: int = 64

    bands_num_in_groups: Tuple[int] = get_sub_bands(sub_band_encoder)[0]
    band_width_in_groups: Tuple[int] = get_sub_bands(sub_band_encoder)[1]

    sub_band_decoder: Dict[str, dict] = {f"decoder{idx}": {"in_features": 64, "out_features": width}
                                         for idx, width in enumerate(band_width_in_groups)}

    dual_path_extension: dict = {
        "num_modules": 3,
        "parameters": {"input_size": 16, "intra_hidden_size": 16, "inter_hidden_size": 16,
                       "groups": 8, "rnn_type": "GRU", "num_layers": 1}
    }

    mag_act: Dict[str, nn.Module] = {"full": nn.ELU, "sub": nn.ReLU}
    last_act: Dict[str, nn.Module] = {"mag_act": nn.ReLU, "pha_act": nn.ELU}


class TrainConfig_48kHz_overlap(FSPENBaseConfig):
    sample_rate: int = 48_000
    n_fft: int = 1024
    hop_length: int = 512
    train_frames: int = 62
    train_points: int = (train_frames - 1) * hop_length
    
    split_last: bool = False
    overlap: bool = False

    full_band_encoder: Dict[str, dict] = {
        "encoder1": {"in_channels": 2, "out_channels": 4, "kernel_size": 6, "stride": 2, "padding": 2,},
        "encoder2": {"in_channels": 4, "out_channels": 16, "kernel_size": 8, "stride": 2, "padding": 3,},
        "encoder3": {"in_channels": 16, "out_channels": 32, "kernel_size": 6, "stride": 2, "padding": 2,},
    }
    full_band_decoder: Dict[str, dict] = {
        "decoder1": {"in_channels": 64, "out_channels": 16, "kernel_size": 6, "stride": 2, "padding": 2,},
        "decoder2": {"in_channels": 32, "out_channels": 4, "kernel_size": 8, "stride": 2, "padding": 3,},
        "decoder3": {"in_channels": 8, "out_channels": 2, "kernel_size": 6, "stride": 2, "padding": 2,}
    }

    sub_band_encoder: Dict[str, dict] = {
        "encoder1": {"group_width": 18, "old_group_width": 18, "conv": {"start_frequency": 0, "end_frequency": 18,
            "in_channels": 1, "out_channels": 32, "kernel_size": 5, "stride": 3, "padding": 1,}}, # 6

        "encoder2": {"group_width": 27, "old_group_width": 18, "conv": {"start_frequency": 9, "end_frequency": 36, 
            "in_channels": 1, "out_channels": 32, "kernel_size": 9, "stride": 4, "padding": 1,}}, # 6

        "encoder3": {"group_width": 57, "old_group_width": 48,
                     "conv": {"start_frequency": 27, "end_frequency": 84, "in_channels": 1, "out_channels": 32, "kernel_size": 14, "stride": 9, "padding": 1,}}, # 6

        "encoder4": {"group_width": 72, "old_group_width": 48,
                     "conv": {"start_frequency": 60, "end_frequency": 132, "in_channels": 1, "out_channels": 32, "kernel_size": 19, "stride": 11, "padding": 1,}}, # 6

        "encoder5": {"group_width": 90, "old_group_width": 66, 
                     "conv": {"start_frequency": 108, "end_frequency": 198, "in_channels": 1, "out_channels": 32, "kernel_size": 32, "stride": 12, "padding": 1,}}, # 6

        "encoder6": {"group_width": 99, "old_group_width": 66,
                      "conv": {"start_frequency": 165, "end_frequency": 264,
                                "in_channels": 1, "out_channels": 32, "kernel_size": 36, "stride": 13, "padding": 1,}}, # 6

        "encoder7": {"group_width": 153, "old_group_width": 120,
                      "conv": {"start_frequency": 231, "end_frequency": 384,
                                "in_channels": 1, "out_channels": 32, "kernel_size": 23, "stride": 12, "padding": 1,}}, # 12

        "encoder8": {"group_width": 189, "old_group_width": 129,
                      "conv": {"start_frequency": 324, "end_frequency": 513,
                                "in_channels": 1, "out_channels": 32, "kernel_size": 20, "stride": 11, "padding": 1,}} # 16
    }

    dummy_params: Dict[str, dict] = {
        "encoder1": {"group_width": 18, "bounds": {"start_frequency": 0, "end_frequency": 18}, "convs": [
            {"in_channels": 1, "out_channels": 32, "kernel_size": 5, "stride": 3, "padding": 1,}]}, # 6

        "encoder2": {"group_width": 18, "bounds": {"start_frequency": 18, "end_frequency": 36}, "convs": [
            {"in_channels": 1, "out_channels": 32, "kernel_size": 5, "stride": 3, "padding": 1, }]}, # 6

        "encoder3": {"group_width": 48, "bounds": {"start_frequency": 36, "end_frequency": 84}, "convs": [
            {"in_channels": 1, "out_channels": 32, "kernel_size": 13, "stride": 7, "padding": 0, }]}, # 6

        "encoder4": {"group_width": 48, "bounds": {"start_frequency": 84, "end_frequency": 132},  "convs": [
            {"in_channels": 1, "out_channels": 32, "kernel_size": 13, "stride": 7, "padding": 0, }]}, # 6

        "encoder5": {"group_width": 66, "bounds": {"start_frequency": 132, "end_frequency": 198}, "convs": [
            {"in_channels": 1, "out_channels": 32, "kernel_size": 16, "stride": 10, "padding": 0, }]}, # 6

        "encoder6": {"group_width": 66, "bounds": {"start_frequency": 198, "end_frequency": 264}, "convs": [
            {"in_channels": 1, "out_channels": 32, "kernel_size": 16, "stride": 10, "padding": 0, }]}, # 6

        "encoder7": {"group_width": 120, "bounds": {"start_frequency": 264, "end_frequency": 384}, "convs": [
            {"in_channels": 1, "out_channels": 32, "kernel_size": 14, "stride": 10, "padding": 2, }]}, # 12

        "encoder8": {"group_width": 129, "bounds": {"start_frequency": 384, "end_frequency": 513}, "convs": [
            {"in_channels": 1, "out_channels": 32, "kernel_size": 23, "stride": 7, "padding": 0, }]}, # 16
    }

    bands_num_in_groups: Tuple[int] = get_sub_bands(sub_band_encoder)[0]
    band_width_in_groups: Tuple[int] = [3, 3, 8, 8, 11, 11, 10, 8]

    sub_band_decoder: Dict[str, dict] = {f"decoder{idx}": {"in_features": 64, "out_features": width}
                                         for idx, width in enumerate(band_width_in_groups)}


    merge_split: dict = {"channels": 128, "bands": 32, "compress_rate": 2}

    num_bands_out: int = 64

    dual_path_extension: dict = {
        "num_modules": 3,
        "parameters": {"input_size": 16, "intra_hidden_size": 16, "inter_hidden_size": 16,
                       "groups": 8, "rnn_type": "GRU", "num_layers": 1}
    }

    mag_act: Dict[str, nn.Module] = {"full": nn.ELU, "sub": nn.ReLU}
    last_act: Dict[str, nn.Module] = {"mag_act": nn.ReLU, "pha_act": nn.ELU}



class TrainConfig_48kHz_enc_ext_lay_1_overlap(FSPENBaseConfig):
    sample_rate: int = 48_000
    n_fft: int = 1024
    hop_length: int = 512
    train_frames: int = 62
    train_points: int = (train_frames - 1) * hop_length

    split_last: bool = True

    full_band_encoder: Dict[str, dict] = {
        "encoder1": {"in_channels": 2, "out_channels": 4, "kernel_size": 6, "stride": 2, "padding": 2, },
        "encoder2": {"in_channels": 4, "out_channels": 16, "kernel_size": 8, "stride": 2, "padding": 3, },
        "encoder3": {"in_channels": 16, "out_channels": 32, "kernel_size": 6, "stride": 2, "padding": 2, },
    }
    full_band_decoder: Dict[str, dict] = {
        "decoder1": {"in_channels": 64, "out_channels": 16, "kernel_size": 6, "stride": 2, "padding": 2, },
        "decoder2": {"in_channels": 32, "out_channels": 4, "kernel_size": 8, "stride": 2, "padding": 3, },
        "decoder3": {"in_channels": 8, "out_channels": 2, "kernel_size": 6, "stride": 2, "padding": 2, },
    }

    sub_band_encoder: Dict[str, dict] = {
        "encoder1": {"group_width": 18, "old_group_width": 18, "bounds": {"start_frequency": 0, "end_frequency": 18},
                     "convs": [
            {"in_channels": 1, "out_channels": 32, "kernel_size": 5, "stride": 3, "padding": 1, }]}, # 6

        "encoder2": {"group_width": 27, "old_group_width": 18, "bounds": {"start_frequency": 9, "end_frequency": 36}, 
                     "convs": [
            {"in_channels": 1, "out_channels": 32, "kernel_size": 9, "stride": 4, "padding": 1, }]}, # 6

        "encoder3": {"group_width": 57, "old_group_width": 48, "bounds": {"start_frequency": 27, "end_frequency": 84},
                     "convs": [
            {"in_channels": 1, "out_channels": 32, "kernel_size": 14, "stride": 9, "padding": 1, }]}, # 6

        "encoder4": {"group_width": 72, "old_group_width": 48, "bounds": {"start_frequency": 60, "end_frequency": 132},
                     "convs": [
            {"in_channels": 1, "out_channels": 32, "kernel_size": 19, "stride": 11, "padding": 1, }]}, # 6

        "encoder5": {"group_width": 90, "old_group_width": 66, "bounds": {"start_frequency": 108, "end_frequency": 198}, 
                     "convs": [
            {"in_channels": 1, "out_channels": 32, "kernel_size": 32, "stride": 12, "padding": 1, }]}, # 6

        "encoder6": {"group_width": 99, "old_group_width": 66, "bounds": {"start_frequency": 165, "end_frequency": 264},
                      "convs": [
            {"in_channels": 1, "out_channels": 32, "kernel_size": 36, "stride": 13, "padding": 1, }]}, # 6

        "encoder7": {"group_width": 153, "old_group_width": 120, "bounds": {"start_frequency": 231, "end_frequency": 384},
                      "convs": [
            {"in_channels": 1, "out_channels": 32, "kernel_size": 23, "stride": 12, "padding": 1, }]}, # 12

        "encoder8": {"group_width": 189, "old_group_width": 129, "bounds": {"start_frequency": 324, "end_frequency": 513},
                      "convs": [
            {"in_channels": 1, "out_channels": 32, "kernel_size": 20, "stride": 11, "padding": 1, }]}, # 16
    }

    dummy_params: Dict[str, dict] = {
        "encoder1": {"group_width": 18, "bounds": {"start_frequency": 0, "end_frequency": 18}, "convs": [
            {"in_channels": 1, "out_channels": 32, "kernel_size": 5, "stride": 3, "padding": 1, }]}, # 6

        "encoder2": {"group_width": 18, "bounds": {"start_frequency": 18, "end_frequency": 36}, "convs": [
            {"in_channels": 1, "out_channels": 32, "kernel_size": 5, "stride": 3, "padding": 1, }]}, # 6

        "encoder3": {"group_width": 48, "bounds": {"start_frequency": 36, "end_frequency": 84}, "convs": [
            {"in_channels": 1, "out_channels": 32, "kernel_size": 13, "stride": 7, "padding": 0, }]}, # 6

        "encoder4": {"group_width": 48, "bounds": {"start_frequency": 84, "end_frequency": 132},  "convs": [
            {"in_channels": 1, "out_channels": 32, "kernel_size": 13, "stride": 7, "padding": 0, }]}, # 6

        "encoder5": {"group_width": 66, "bounds": {"start_frequency": 132, "end_frequency": 198}, "convs": [
            {"in_channels": 1, "out_channels": 32, "kernel_size": 16, "stride": 10, "padding": 0, }]}, # 6

        "encoder6": {"group_width": 66, "bounds": {"start_frequency": 198, "end_frequency": 264}, "convs": [
            {"in_channels": 1, "out_channels": 32, "kernel_size": 16, "stride": 10, "padding": 0, }]}, # 6

        "encoder7": {"group_width": 120, "bounds": {"start_frequency": 264, "end_frequency": 384}, "convs": [
            {"in_channels": 1, "out_channels": 32, "kernel_size": 14, "stride": 10, "padding": 2, }]}, # 12

        "encoder8": {"group_width": 129, "bounds": {"start_frequency": 384, "end_frequency": 513}, "convs": [
            {"in_channels": 1, "out_channels": 32, "kernel_size": 23, "stride": 7, "padding": 0, }]}, # 16
    }

    overlap: bool = True

    sub_band_decoder: Dict[str, dict] = build_sub_band_decoder_params(dummy_params, [6, 6, 6, 6, 6, 6, 12, 16])

    merge_split: dict = {"channels": 128, "bands": 32, "compress_rate": 2}

    num_bands_out: int = 64

    dual_path_extension: dict = {
        "num_modules": 3,
        "parameters": {"input_size": 16, "intra_hidden_size": 16, "inter_hidden_size": 16,
                       "groups": 8, "rnn_type": "GRU", "num_layers": 1}
    }

    mag_act: Dict[str, nn.Module] = {"full": nn.ELU, "sub": nn.ELU}
    last_act: Dict[str, nn.Module] = {"mag_act": nn.Sigmoid, "pha_act": nn.Tanh}

            
class TrainConfig_48kHz_enc_ext(FSPENBaseConfig):
    sample_rate: int = 48_000
    n_fft: int = 1024
    hop_length: int = 512
    train_frames: int = 62
    train_points: int = (train_frames - 1) * hop_length

    split_last: bool = True
    overlap: bool = False

    full_band_encoder: Dict[str, dict] = {
        "encoder1": {"in_channels": 2, "out_channels": 4, "kernel_size": 6, "stride": 2, "padding": 2, },
        "encoder2": {"in_channels": 4, "out_channels": 16, "kernel_size": 8, "stride": 2, "padding": 3, },
        "encoder3": {"in_channels": 16, "out_channels": 32, "kernel_size": 6, "stride": 2, "padding": 2, },
    }

    full_band_decoder: Dict[str, dict] = {
        "decoder1": {"in_channels": 64, "out_channels": 16, "kernel_size": 6, "stride": 2, "padding": 2, },
        "decoder2": {"in_channels": 32, "out_channels": 4, "kernel_size": 8, "stride": 2, "padding": 3, },
        "decoder3": {"in_channels": 8, "out_channels": 2, "kernel_size": 6, "stride": 2, "padding": 2,}
    }

    sub_band_encoder: Dict[str, dict] = {
        "encoder1": {"group_width": 18, "bounds": {"start_frequency": 0, "end_frequency": 18}, "convs": [
            {"in_channels": 1, "out_channels": 8, "kernel_size": 4, "stride": 2, "padding": 1, },
            {"in_channels": 8, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 0, },
            {"in_channels": 16, "out_channels": 32, "kernel_size": 2, "stride": 1, "padding": 0, }]}, # 6

        "encoder2": {"group_width": 18, "bounds": {"start_frequency": 18, "end_frequency": 36}, "convs": [
            {"in_channels": 1, "out_channels": 8, "kernel_size": 4, "stride": 2, "padding": 1, },
            {"in_channels": 8, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 0, },
            {"in_channels": 16, "out_channels": 32, "kernel_size": 2, "stride": 1, "padding": 0, }]}, # 6

        "encoder3": {"group_width": 48, "bounds": {"start_frequency": 36, "end_frequency": 84}, "convs": [
            {"in_channels": 1, "out_channels": 8, "kernel_size": 6, "stride": 3, "padding": 0, },
            {"in_channels": 8, "out_channels": 16, "kernel_size": 3, "stride": 2, "padding": 0, },
            {"in_channels": 16, "out_channels": 32, "kernel_size": 2, "stride": 1, "padding": 0, }]}, # 6

        "encoder4": {"group_width": 48, "bounds": {"start_frequency": 84, "end_frequency": 132},  "convs": [
            {"in_channels": 1, "out_channels": 8, "kernel_size": 6, "stride": 3, "padding": 0, },
            {"in_channels": 8, "out_channels": 16, "kernel_size": 3, "stride": 2, "padding": 0, },
            {"in_channels": 16, "out_channels": 32, "kernel_size": 2, "stride": 1, "padding": 0, }]}, # 6

        "encoder5": {"group_width": 66, "bounds": {"start_frequency": 132, "end_frequency": 198}, "convs": [
            {"in_channels": 1, "out_channels": 8, "kernel_size": 6, "stride": 4, "padding": 0, }, 
            {"in_channels": 8, "out_channels": 16, "kernel_size": 4, "stride": 2, "padding": 0, },
            {"in_channels": 16, "out_channels": 32, "kernel_size": 2, "stride": 1, "padding": 0, }]}, # 6

        "encoder6": {"group_width": 66, "bounds": {"start_frequency": 198, "end_frequency": 264}, "convs": [
            {"in_channels": 1, "out_channels": 8, "kernel_size": 6, "stride": 4, "padding": 0, }, 
            {"in_channels": 8, "out_channels": 16, "kernel_size": 4, "stride": 2, "padding": 0, },
            {"in_channels": 16, "out_channels": 32, "kernel_size": 2, "stride": 1, "padding": 0, }]}, # 6

        "encoder7": {"group_width": 120, "bounds": {"start_frequency": 264, "end_frequency": 384}, "convs": [
            {"in_channels": 1, "out_channels": 8, "kernel_size": 6, "stride": 2, "padding": 0, },
            {"in_channels": 8, "out_channels": 16, "kernel_size": 6, "stride": 2, "padding": 0, },
            {"in_channels": 16, "out_channels": 32, "kernel_size": 5, "stride": 2, "padding": 0, }]}, # 12

        "encoder8": {"group_width": 129, "bounds": {"start_frequency": 384, "end_frequency": 513}, "convs": [
            {"in_channels": 1, "out_channels": 8, "kernel_size": 6, "stride": 2, "padding": 2, },
            {"in_channels": 8, "out_channels": 16, "kernel_size": 4, "stride": 2, "padding": 0, },
            {"in_channels": 16, "out_channels": 32, "kernel_size": 3, "stride": 2, "padding": 1, }]}, # 16
    }

    sub_band_decoder: Dict[str, dict] = build_sub_band_decoder_params(sub_band_encoder, [6, 6, 6, 6, 6, 6, 12, 16])

    merge_split: dict = {"channels": 128, "bands": 32, "compress_rate": 2}

    num_bands_out: int = 64

    dual_path_extension: dict = {
        "num_modules": 3,
        "parameters": {"input_size": 16, "intra_hidden_size": 16, "inter_hidden_size": 16,
                       "groups": 8, "rnn_type": "GRU", "num_layers": 1}
    }

    mag_act: Dict[str, nn.Module] = {"full": nn.ELU, "sub": nn.ELU}
    last_act: Dict[str, nn.Module] = {"mag_act": nn.Sigmoid, "pha_act": nn.Tanh}
