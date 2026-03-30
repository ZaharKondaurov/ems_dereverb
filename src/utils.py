import numpy as np
import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from termcolor import colored
from collections import defaultdict


PCS = {        # Perceptual Contrast Stretching
    (0, 3): 1,
    (3, 6): 1.070175439,
    (6, 9): 1.182456140,
    (9, 12): 1.287719298,
    (12, 138): 1.4,
    (138, 166): 1.322807018,
    (166, 200): 1.238596491,
    (200, 241): 1.161403509,
    (241, 257): 1.077192982
}

def beautiful_int(i):
    i = str(i)
    return ".".join(reversed([i[max(j, 0):j+3] for j in range(len(i) - 3, -3, -3)]))

def vorbis_window(winlen, device="cuda"):
    sq = torch.sin(torch.pi/2*(torch.sin(torch.pi/winlen*(torch.arange(winlen)-0.5))**2)).float()
    return sq


def create_warmup_cosine_scheduler(
    optimizer,
    warmup_epochs: int,
    total_epochs: int,
    warmup_start_lr: float = 1e-6,
    base_lr: float = 1e-3,
    eta_min: float = 0.0
):
    """
    Создает scheduler: линейный warmup до base_lr, затем cosine annealing.
    
    Args:
        optimizer: Оптимизатор
        warmup_epochs: Кол-во эпох для warmup
        total_epochs: Общее кол-во эпох
        warmup_start_lr: Начальный LR для warmup
        base_lr: Пиковый LR (начало cosine)
        eta_min: Минимальный LR для cosine
    """
    # LinearLR: start_factor умножается на initial_lr оптимизатора
    # Устанавливаем initial_lr = base_lr, start_factor = warmup_start_lr / base_lr
    scheduler_warmup = LinearLR(
        optimizer, 
        start_factor=warmup_start_lr / base_lr, 
        total_iters=warmup_epochs
    )
    
    scheduler_cosine = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=eta_min
    )
    
    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_cosine],
        milestones=[warmup_epochs]
    )
    
    return scheduler


def mag_phase_stft(y, n_fft, hop_size, win_size, compress_factor=1.0, center=True, addeps=False):
    """
    Compute magnitude and phase using STFT.

    Args:
        y (torch.Tensor): Input audio signal.
        n_fft (int): FFT size.
        hop_size (int): Hop size.
        win_size (int): Window size.
        compress_factor (float, optional): Magnitude compression factor. Defaults to 1.0.
        center (bool, optional): Whether to center the signal before padding. Defaults to True.
        eps (bool, optional): Whether adding epsilon to magnitude and phase or not. Defaults to False. 

    Returns:
        tuple: Magnitude, phase, and complex representation of the STFT.
    """
    eps = 1e-10
    window = vorbis_window(n_fft).to(y.device)
    stft_spec = torch.stft(
                    y, n_fft, 
                    hop_length=hop_size, 
                    win_length=win_size, 
                    window=window,
                    center=center, 
                    pad_mode='reflect', 
                    normalized=True, 
                    return_complex=True)

    if addeps==False:
        mag = torch.abs(stft_spec)
        pha = torch.angle(stft_spec)
    else:
        real_part = stft_spec.real
        imag_part = stft_spec.imag
        mag = torch.sqrt(real_part.pow(2) + imag_part.pow(2) + eps)
        pha = torch.atan2(imag_part + eps, real_part + eps)
    
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)
    return com

def use_pcs(spec, n_fft=512):
    k = n_fft // 512
    pcs = torch.ones(n_fft // 2 + 1, device=spec.device)
    for (start, end), gamma in PCS.items():
        pcs[start * k: end * k] = gamma

    spec_mag = torch.log1p(spec.abs())
    spec_mag_pcs = pcs * spec_mag.transpose(-2, -1)

    spec_pcs = torch.polar(spec_mag_pcs.transpose(-2, -1), spec.angle())

    return spec_pcs

def inv_pcs(mag, phase):
    spec_mag = torch.expm1(mag)
    return torch.polar(spec_mag, phase)

# Считаем общее число параметров в нашей модели
def model_num_params(model, verbose_all=True, verbose_only_learnable=False):
    sum_params = 0
    sum_learnable_params = 0
    submodules = defaultdict(lambda : [0, 0])
    for name, param in model.named_parameters():
        num_params = np.prod(param.shape)
        if verbose_all or (verbose_only_learnable and param[1].requires_grad):
            print(
                colored(
                    '{: <42} ~  {: <9} params ~ grad: {}'.format(
                        name,
                        beautiful_int(num_params),
                        param.requires_grad,
                    ),
                    {True: "green", False: "red"}[param.requires_grad],
                )
            )
        sum_params += num_params
        sm = name.split(".")[0]
        submodules[sm][0] += num_params
        if param.requires_grad:
            sum_learnable_params += num_params
            submodules[sm][1] += num_params
    print(
        f'\nIn total:\n  - {beautiful_int(sum_params)} params\n  - {beautiful_int(sum_learnable_params)} learnable params'
    )

    for sm, v in submodules.items():
        print(
            f"\n . {sm}:\n .   - {beautiful_int(submodules[sm][0])} params\n .   - {beautiful_int(submodules[sm][1])} learnable params"
        )
    return sum_params, sum_learnable_params

def model_eval(model, input_spec, configs, device="cpu", hid_size=64):
    # input_spec = input_spec.to(device)

    abs_spectrum = input_spec.abs()
    # input_spec_ = torch.permute(torch.view_as_real(input_spec), dims=(0, 2, 3, 1))
    input_spec_ = torch.stack((input_spec.abs(), input_spec.angle()), dim=-1).permute((0, 2, 3, 1))

    batch, frames, channels, frequency = input_spec_.shape
    abs_spectrum = torch.permute(abs_spectrum, dims=(0, 2, 1))
    abs_spectrum = torch.reshape(abs_spectrum, shape=(batch, frames, 1, frequency))
    h0 = [[torch.zeros(configs.dual_path_extension["parameters"]["num_layers"], batch * configs.num_bands_out, configs.dual_path_extension["parameters"]["inter_hidden_size"], device=input_spec.device) for _ in range(8)] for _ in range(configs.dual_path_extension["num_modules"])]

    assert torch.isnan(input_spec_).any().item() is False, "input_spec has NaNs"
    assert torch.isnan(abs_spectrum).any().item() is False, "abs_spectrum has NaNs"

    output, hid_out = model(input_spec_, abs_spectrum, h0)
    # print(output.shape, input_spec.angle().shape)
    # output = torch.concat([output, input_spec.angle()])

    # output = torch.permute(output, dims=(0, 3, 1, 2))
    # output = torch.concat([output, input_spec.angle()[..., None]], dim=-1)
    # output = torch.polar(output.contiguous()[..., 0], input_spec.angle())
    # output = torch.view_as_complex(output.contiguous())
    output = torch.permute(output, dims=(0, 3, 1, 2))
    assert torch.isnan(output[..., 0]).any().item() is False, "abs output has NaNs"
    assert torch.isnan(output[..., 1]).any().item() is False, "phase output has NaNs"

    output = torch.polar(output[..., 0], output[..., 1])

    return output, hid_out

def model_eval_old(model, input_spec, configs, device="cpu", hid_size=64):
    # input_spec = input_spec.to(device)

    abs_spectrum = input_spec.abs()
    input_spec_ = torch.permute(torch.view_as_real(input_spec), dims=(0, 2, 3, 1))
    # input_spec_ = torch.stack((input_spec.abs(), input_spec.angle()), dim=-1).permute((0, 2, 3, 1))

    batch, frames, channels, frequency = input_spec_.shape
    abs_spectrum = torch.permute(abs_spectrum, dims=(0, 2, 1))
    abs_spectrum = torch.reshape(abs_spectrum, shape=(batch, frames, 1, frequency))
    h0 = [[torch.zeros(configs.dual_path_extension["parameters"]["num_layers"], batch * hid_size, configs.dual_path_extension["parameters"]["inter_hidden_size"], device=input_spec.device) for _ in range(8)] for _ in range(configs.dual_path_extension["num_modules"])]

    output, abs_addon, hid_out = model(input_spec_, abs_spectrum, h0)
    # print(output.shape, input_spec.angle().shape)
    # output = torch.concat([output, input_spec.angle()])

    output = torch.permute(output, dims=(0, 3, 1, 2))
    # output = torch.concat([output, input_spec.angle()[..., None]], dim=-1)
    # output = torch.polar(output.contiguous()[..., 0], input_spec.angle())
    output = torch.view_as_complex(output.contiguous())
    out_abs, out_pha = output.abs(), output.angle()
    abs_addon = torch.permute(abs_addon, dims=(0, 3, 1, 2))
    # print(out_abs.shape, abs_addon.shape)
    out_abs += abs_addon[..., -1]
    # output = torch.permute(output, dims=(0, 3, 1, 2))
    result = torch.polar(out_abs, out_pha)


    return result, hid_out

def model_eval_3_heads(model, input_spec, configs, device="cpu", hid_size=64):
    # input_spec = input_spec.to(device)

    abs_spectrum = input_spec.abs()
    input_spec_ = torch.stack((input_spec.abs(), input_spec.angle()), dim=-1).permute((0, 2, 3, 1))

    batch, frames, channels, frequency = input_spec_.shape
    abs_spectrum = torch.permute(abs_spectrum, dims=(0, 2, 1))
    abs_spectrum = torch.reshape(abs_spectrum, shape=(batch, frames, 1, frequency))
    h0 = [[torch.zeros(1, batch * hid_size, 16, device=input_spec.device) for _ in range(8)] for _ in range(configs.dual_path_extension["num_modules"])]

    output, hid_out = model(input_spec_, abs_spectrum, h0)
    
    output = torch.permute(output, dims=(0, 3, 1, 2))

    output_signal = torch.polar(output[..., 0], output[..., 1])
    output_noise = torch.polar(output[..., 2], output[..., 3])
    output_rir = torch.polar(output[...,4], output[..., 5])

    return output_signal, output_noise, output_rir, hid_out

def model_eval_fspen2x_ver3(model, input_spec, device="cpu", hid_size=64):
    # input_spec = input_spec.to(device)

    abs_spectrum = input_spec.abs()
    input_spec_ = torch.permute(torch.view_as_real(input_spec), dims=(0, 2, 3, 1))
    batch, frames, channels, frequency = input_spec_.shape
    abs_spectrum = torch.permute(abs_spectrum, dims=(0, 2, 1))
    abs_spectrum = torch.reshape(abs_spectrum, shape=(batch, frames, 1, frequency))
    h0 = [[torch.zeros(1, batch * hid_size, 8, device=input_spec.device) for _ in range(8)] for _ in range(3)]

    output, hid_out = model(input_spec_, abs_spectrum, h0)

    output = torch.permute(output, dims=(0, 3, 1, 2))
    output = torch.view_as_complex(output)

    return output, hid_out
