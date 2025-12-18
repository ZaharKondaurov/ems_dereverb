import os
from abc import ABC, abstractmethod

import numpy as np
from scipy.signal import fftconvolve
from scipy.io import wavfile

import torch
import torchaudio
from torchaudio.transforms import Resample
from torch.utils.data import Dataset

import librosa
import pyroomacoustics as pra

from typing import Union, Tuple, Callable, List, Dict

from random import shuffle, randint, choice, uniform

WALLS_KEYWORDS = ["hard_surface", "ceramic_tiles", "plasterboard", "wooden_lining", "glass_3mm"]    # Убрать материалы
FLOOR_KEYWORDS = ["linoleum_on_concrete", "carpet_cotton"]
CEILING_KEYWORDS = ["ceiling_plasterboard", "ceiling_fissured_tile", "ceiling_metal_panel", ]


class SignalDataset(ABC, Dataset):

    def __init__(self, data_dir_path: str, sr: int = 16_000,
                 snr: Union[int, Tuple[int, int], List[int], Dict[int, List[int]]] = 0,
                 chunk_size: int = 16_000 * 2,
                 stride: int = 16_000,
                 noise_dir: str = None,
                 rir_dir: Union[str, Dict[int, str]] = None,
                 rir_target: bool = False,
                 room_square: Tuple[float, float] = (7., 14.),
                 room_height: Tuple[float, float] = (3., 4.),
                 return_noise: bool = False,
                 return_rir: bool = False,
                 max_seq_len: int = None,
                 partition: int = None,
                 noise_proba: float = 1.0,
                 rir_proba: float = 1.0,
                 mode="train"):

        self.path = data_dir_path
        # self.signal_files = [x for x in os.listdir(self.path) if x[-3:] == "wav"]
        self.signal_files = [os.path.join(r, f) for r, d, fs in os.walk(self.path) for f in fs if f[-3:] == "wav"] #  [x for x in os.listdir(self.path) if x[-3:] == "wav"]
        if partition is not None:
            self.signal_files = self.signal_files[:partition]
        shuffle(self.signal_files)

        self.sr = sr
        self.snr = snr
        self.snr_dict = None
        if isinstance(snr, dict):
            self.snr_dict = snr
            self.snr = list(self.snr_dict.values())[0]

        self.chunk_size = chunk_size
        self.stride = stride
        self.room_square = room_square
        self.room_height = room_height

        self.noise_dir = noise_dir

        if self.noise_dir is not None:
            self.noise_files = [os.path.join(r, f) for r, d, fs in os.walk(self.noise_dir) for f in fs if f[-3:] == "wav"] # os.listdir(noise_dir)
            shuffle(self.noise_files)

        self.rir_dir = rir_dir
        # self.rir_dir_target = rir_dir_target

        self.rir_dict = None
        # self.rir_dict_target = None
        if isinstance(rir_dir, dict):
            self.rir_dict = rir_dir
            self.rir_dir = list(self.rir_dict.values())[0]
            # if rir_dir_target is not None:
            #     self.rir_dict_target = rir_dir_target
            #     self.rir_dir_target = list(self.rir_dict.values())[0]
        self.rir_target = rir_target
        if self.rir_dir is not None:
            self.rir_files = [os.path.join(r, f) for r, d, fs in os.walk(self.rir_dir) for f in fs if f[-3:] == "wav"] # os.listdir(rir_dir)
            shuffle(self.rir_files)
            # if rir_target:
            #     self.rir_files_target = [os.path.join(r + "_target", f) for r, d, fs in os.walk(self.rir_dir) for f in fs if f[-3:] == "wav"]

            # if self.rir_dir_target is not None:
            #     self.rir_files_target = [os.path.join(r, f) for r, d, fs in os.walk(self.rir_dir_target) for f in fs if f[-3:] == "wav"]

        self.return_noise = return_noise
        self.return_rir = return_rir
        self.max_seq_len = max_seq_len

        self.noise_proba = noise_proba
        self.rir_proba = rir_proba
        self.epoch = 0

    # @staticmethod
    # def simulate_noise(signal: torch.Tensor, noise: torch.Tensor, snr_db: int) -> torch.Tensor:
    #     len_noise = noise.shape[-1]
    #     noise_cat = [noise]
    #     while len_noise < signal.shape[-1]:
    #         len_noise += noise.shape[-1]
    #         noise_cat.append(noise)
    #     noise = torch.cat(noise_cat, dim=-1)
    #
    #     if noise.shape[-1] > signal.shape[-1]:
    #         noise = noise[..., :signal.shape[-1]]
    #
    #     target_snr = (10 ** (snr_db / 10))
    #     noise_power = torch.mean(signal ** 2) / target_snr
    #     noise_mult = torch.sqrt(noise_power / torch.mean(noise ** 2))
    #     abs_max = noise_mult * torch.abs(noise).max().item()
    #     if abs_max > 1:
    #         noise_mult /= abs_max
    #     # rms_signal = torch.sqrt(torch.mean(signal ** 2))
    #     # rms_noise = torch.sqrt(torch.mean(noise ** 2))
    #     #
    #     # target_rms_noise = rms_signal / (10 ** (snr_db / 10))
    #
    #     noise = noise * noise_mult # (target_rms_noise / rms_noise)
    #     # print(noise_mult)
    #     return noise

    @staticmethod
    def to_db(ratio):
        assert ratio >= 0
        ratio_db = 10. * np.log10(ratio + 1e-8)
        return ratio_db

    @staticmethod
    def from_db(ratio_db):
        ratio = 10 ** (ratio_db / 10.) - 1e-8
        return ratio

    @staticmethod
    def simulate_noise(src_audio, ns_audio, snr):
        if ns_audio.shape[-1] < src_audio.shape[-1]:
            ns_audio = torch.tile(ns_audio, (1, int(np.ceil(src_audio.shape[-1] / ns_audio.shape[-1]))))
        ns_audio = ns_audio[..., :src_audio.shape[-1]]

        try:
            target_snr_n = SignalDataset.from_db(snr)
            ns_target_sq = torch.mean(src_audio ** 2, dim=-1) / target_snr_n
            ns_mult = torch.sqrt(ns_target_sq / torch.mean(ns_audio ** 2, dim=-1))
        except Exception:
            print('Failed!')
            ns_mult = 1.
        abs_max = ns_mult * torch.abs(ns_audio).max().item()
        if abs_max > 1.:
            ns_mult /= abs_max
        ns_mult = ns_mult.item()
        # print(ns_mult, ns_audio.shape)
        return ns_mult * ns_audio
    
    def set_epoch(self, epoch: int):
        self.epoch = epoch

        if self.snr_dict is not None:
            for step, snr in self.snr_dict.items():
                if epoch >= step:
                    self.snr = snr
                else:
                    break
        
        if self.rir_dict is not None:
            self.rir_files = []
            for step, rir_path in self.rir_dict.items():
                if epoch >= step:
                    self.rir_files.extend([os.path.join(r, f) for r, d, fs in os.walk(rir_path) for f in fs if f[-3:] == "wav"])
                    # if self.rir_target:
                    #     self.rir_files_target.extend([os.path.join(r + "_target", f) for r, d, fs in os.walk(rir_path) for f in fs if f[-3:] == "wav"])
                else:
                    break
            shuffle(self.rir_files)

    def simulate_rir_shoebox(self, signal: torch.Tensor) -> torch.Tensor:
        square = uniform(*self.room_square)
        width = uniform(2.5, square * 0.75)
        length = square / width
        height = uniform(*self.room_height)

        rt60 = uniform(0.3, 1.25)   # Делать длинный ревёрб
        room_dim = [length, width, height]

        e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)

        wall = pra.Material(choice(WALLS_KEYWORDS))
        ceil = pra.Material(choice(CEILING_KEYWORDS))
        floor = pra.Material(choice(FLOOR_KEYWORDS))

        material = {"east": wall, "west": wall, "north": wall, "south": wall, "ceiling": ceil, "floor": floor}

        room = pra.ShoeBox(room_dim, fs=self.sr, materials=material, max_order=max_order,
                           use_rand_ism=True, max_rand_disp=0.05, ray_tracing=False)

        source_locs = [uniform(0.01, length), uniform(0.01, width), uniform(1.0, 2.0)]
        mic_locs = np.array([x * 0.98 for x in source_locs])[:, None]

        room.add_source(source_locs, signal=signal.squeeze(), delay=0.5)

        room.add_microphone_array(mic_locs)
        room.compute_rir()
        room.simulate()     # Внутри есть параметр snr, возможно, он пригодится

        return room.rir[0][0]   # [микрофон, источник]

    @abstractmethod
    def _preprocess(self, signal: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __len__(self):
        return len(self.signal_files)

    @staticmethod
    def normalize_audio(target_signal, signal=None):
        if (signal is not None) and torch.max(torch.abs(signal)) > 0:
            scale = torch.max(torch.abs(signal))
            target_signal = target_signal / scale
            signal = signal / scale

        if torch.max(torch.abs(target_signal)) > 0:
            target_signal = target_signal / torch.max(torch.abs(target_signal))

        return target_signal, signal

    def __getitem__(self, idx):
        if isinstance(self.snr, tuple):
            snr_db = randint(self.snr[0], self.snr[1])
        elif isinstance(self.snr, int):
            snr_db = self.snr
        elif isinstance(self.snr, list):
            snr_db = choice(self.snr)
        else:
            assert "Invalid snr!"

        filename = self.signal_files[idx]

        target_signal, signal_sr = torchaudio.load(filename)
        # print(target_signal.shape)
        noise = None
        rir_component = None
        target_signal, _ = SignalDataset.normalize_audio(target_signal)
        if signal_sr != self.sr:
            resampler = Resample(signal_sr, self.sr)
            target_signal = resampler(target_signal)

        if self.rir_dir is not None and uniform(0, 1) < self.rir_proba:
            filename_rir = choice(self.rir_files)
            rir, rir_sr = torchaudio.load(filename_rir)
            if rir.shape[0] > 1:
                rir = torch.from_numpy(librosa.to_mono(rir.numpy()))[None, :]
            if rir_sr != self.sr:
                resampler = Resample(rir_sr, self.sr)
                rir = resampler(rir)

            rir, _ = SignalDataset.normalize_audio(rir)

            rir_signal = torch.from_numpy(fftconvolve(target_signal, rir, mode='full', axes=-1))
            rir_signal = rir_signal[..., :target_signal.shape[-1]]

            if self.rir_target:
                # rir_basename = os.path.basename(filename_rir)
                rir_directory, rir_basename = os.path.split(filename_rir)
                dir_, dir_name = os.path.split(rir_directory)
                if "soft" not in dir_name:
                    targer_rir_path = os.path.join(dir_, dir_name + "_target", rir_basename)

                    target_rir, target_rir_sr = torchaudio.load(targer_rir_path)

                    if target_rir.shape[0] > 1:
                        target_rir = torch.from_numpy(librosa.to_mono(target_rir.numpy()))[None, :]
                    if target_rir_sr != self.sr:
                        resampler = Resample(target_rir_sr, self.sr)
                        target_rir = resampler(target_rir)

                    target_rir, _ = SignalDataset.normalize_audio(target_rir)
                    
                    target_signal = torch.from_numpy(fftconvolve(target_signal, target_rir, mode='full', axes=-1))[..., :target_signal.shape[-1]]

            rir_component = rir_signal - target_signal
        else:
            rir_signal = target_signal

        if self.noise_dir is not None and uniform(0, 1) < self.noise_proba:
            filename_noise = choice(self.noise_files)
            noise, noise_sr = torchaudio.load(filename_noise)
            # print(noise.shape)
            if noise.shape[0] > 1:
                noise = torch.from_numpy(librosa.to_mono(noise.numpy()))[None, :]
            # print(noise.shape)
            if noise_sr != self.sr:
                resampler = Resample(noise_sr, self.sr)
                noise = resampler(noise)

            # noise -= noise.mean(dim=1)
            noise, _ = SignalDataset.normalize_audio(noise)
            noise = self.simulate_noise(rir_signal, noise, snr_db)
            # print(noise.shape, rir_signal.shape)
            output = rir_signal + noise
        else:
            output = rir_signal

        target_signal, output = SignalDataset.normalize_audio(target_signal, output)

        if noise is not None:
            noise, _ = SignalDataset.normalize_audio(noise)
        if rir_component is not None:
            rir_component, _ = SignalDataset.normalize_audio(rir_component)

        if self.max_seq_len is not None:
            output_padded = torch.zeros(1, self.max_seq_len)
            output_padded[..., :output.shape[-1]] = output[..., :self.max_seq_len]

            target_padded = torch.zeros(1, self.max_seq_len)
            target_padded[..., :target_signal.shape[-1]] = target_signal[..., :self.max_seq_len]

            rir_padded = None
            if (self.rir_dir is not None) and self.return_rir:
                rir_padded = torch.zeros(1, self.max_seq_len)
                rir_padded[..., :rir_component.shape[-1]] = rir_component[..., :self.max_seq_len]

            noise_padded = None
            if (self.noise_dir is not None) and self.return_noise:
                noise_padded = torch.zeros(1, self.max_seq_len)
                noise_padded[..., :noise.shape[-1]] = noise[..., :self.max_seq_len]

            return output_padded, target_padded, noise_padded, rir_padded

        return output, target_signal, noise if self.return_noise else None, rir_component if self.return_rir else None


class TRUNetDataset(SignalDataset):

    def __init__(self, data_dir_path: str, sr: int = 16_000,
                 snr: Union[int, Tuple[int, int], List[int], Dict[int, List[int]]] = 0,
                 chunk_size: int = 16_000 * 2,
                 stride: int = 16_000,
                 noise_dir: str = None,
                 rir_dir: Union[str, Dict[int, str]] = None,
                 rir_target: bool = False,
                 room_square: Tuple[float, float] = (7., 14.),
                 room_height: Tuple[float, float] = (3., 4.),
                 return_noise: bool = False,
                 return_rir: bool = False,
                 max_seq_len: int = None,
                 partition: int = None,
                 noise_proba: float = 1.0,
                 rir_proba: float = 1.0,
                 mode="train",):

        super(TRUNetDataset, self).__init__(data_dir_path=data_dir_path, sr=sr, snr=snr, chunk_size=chunk_size,
                                            stride=stride, noise_dir=noise_dir, rir_dir=rir_dir, rir_target=rir_target, 
                                            room_square=room_square,
                                            room_height=room_height, return_noise=return_noise,
                                            return_rir=return_rir, max_seq_len=max_seq_len, partition=partition,
                                            noise_proba=noise_proba, rir_proba=rir_proba, mode=mode)

    @staticmethod
    def _preprocess(signal: torch.Tensor) -> torch.Tensor:
        return signal


class VoiceBankDataset(ABC, Dataset):

    def __init__(self,
                 noise_dir_path: str,
                 clean_dir_path: str,
                 sr: int = 16_000,
                 chunk_size: int = 16_000 * 2,
                 stride: int = 16_000,
                 max_seq_len: int = None,
                 partition: int = None,
                 mode="train"):

        self.noise_path = noise_dir_path
        self.clean_dir_path = clean_dir_path
        self.signal_files = [x for x in os.listdir(self.noise_path) if x[-3:] == "wav"]
        if partition is not None:
            self.signal_files = self.signal_files[:partition]
        shuffle(self.signal_files)

        self.sr = sr
        self.chunk_size = chunk_size
        self.stride = stride
        self.max_seq_len = max_seq_len

    # @abstractmethod
    # def _preprocess(self, signal: torch.Tensor) -> torch.Tensor:
    #     raise signal

    def __len__(self):
        return len(self.signal_files)

    @staticmethod
    def normalize_audio(target_signal, signal=None):
        if (signal is not None) and torch.max(torch.abs(signal)) > 0:
            scale = torch.max(torch.abs(signal))
            target_signal = target_signal / scale
            signal = signal / scale

        if torch.max(torch.abs(target_signal)) > 0:
            target_signal = target_signal / torch.max(torch.abs(target_signal))

        return target_signal, signal

    def __getitem__(self, idx):
        filename = self.signal_files[idx]

        try:
            target_signal, target_sr = torchaudio.load(os.path.join(self.clean_dir_path, filename))
        except FileNotFoundError:
            assert f"There is not {filename} in clean dir"

        try:
            noise_signal, noise_sr = torchaudio.load(os.path.join(self.noise_path, filename))
        except FileNotFoundError:
            assert f"There is not {filename} in noise dir"

        if target_sr != self.sr:
            resampler = Resample(target_sr, self.sr)
            target_signal = resampler(target_signal)

        if noise_sr != self.sr:
            resampler = Resample(noise_sr, self.sr)
            noise_signal = resampler(noise_signal)

        target_signal, noise_signal = SignalDataset.normalize_audio(target_signal, noise_signal)

        return noise_signal, target_signal
