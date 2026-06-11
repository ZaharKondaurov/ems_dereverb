import numpy as np
import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from termcolor import colored
from collections import defaultdict


def beautiful_int(i):
    i = str(i)
    return ".".join(reversed([i[max(j, 0) : j + 3] for j in range(len(i) - 3, -3, -3)]))


def vorbis_window(winlen):
    sq = torch.sin(
        torch.pi
        / 2
        * (torch.sin(torch.pi / winlen * (torch.arange(winlen) - 0.5)) ** 2)
    ).float()
    return sq


def create_warmup_cosine_scheduler(
    optimizer,
    warmup_epochs: int,
    total_epochs: int,
    warmup_start_lr: float = 1e-6,
    base_lr: float = 1e-3,
    eta_min: float = 0.0,
):
    scheduler_warmup = LinearLR(
        optimizer, start_factor=warmup_start_lr / base_lr, total_iters=warmup_epochs
    )

    scheduler_cosine = CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs, eta_min=eta_min
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_cosine],
        milestones=[warmup_epochs],
    )

    return scheduler


def model_num_params(model, verbose_all=True, verbose_only_learnable=False):
    sum_params = 0
    sum_learnable_params = 0
    submodules = defaultdict(lambda: [0, 0])
    for name, param in model.named_parameters():
        num_params = np.prod(param.shape)
        if verbose_all or (verbose_only_learnable and param[1].requires_grad):
            print(
                colored(
                    "{: <42} ~  {: <9} params ~ grad: {}".format(
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
        f"\nIn total:\n  - {beautiful_int(sum_params)} params\n  - {beautiful_int(sum_learnable_params)} learnable params"
    )

    for sm, v in submodules.items():
        print(
            f"\n . {sm}:\n .   - {beautiful_int(submodules[sm][0])} params\n .   - {beautiful_int(submodules[sm][1])} learnable params"
        )
    return sum_params, sum_learnable_params


def model_eval(model, input_spec, configs, h0=None):
    abs_spectrum = input_spec.abs()
    input_spec_ = torch.stack((input_spec.abs(), input_spec.angle()), dim=-1).permute(
        (0, 2, 3, 1)
    )

    batch, frames, channels, frequency = input_spec_.shape
    abs_spectrum = torch.permute(abs_spectrum, dims=(0, 2, 1))
    abs_spectrum = torch.reshape(abs_spectrum, shape=(batch, frames, 1, frequency))
    if h0 is None:
        h0 = [
            [
                torch.zeros(
                    configs.dual_path_extension["parameters"]["num_layers"],
                    batch * configs.num_bands_out,
                    configs.dual_path_extension["parameters"]["inter_hidden_size"],
                    device=input_spec.device,
                )
                for _ in range(8)
            ]
            for _ in range(configs.dual_path_extension["num_modules"])
        ]

    assert torch.isnan(input_spec_).any().item() is False, "input_spec has NaNs"
    assert torch.isnan(abs_spectrum).any().item() is False, "abs_spectrum has NaNs"

    output, hid_out = model(input_spec_, abs_spectrum, h0)

    output = torch.permute(output, dims=(0, 3, 1, 2))
    assert torch.isnan(output[..., 0]).any().item() is False, "abs output has NaNs"
    assert torch.isnan(output[..., 1]).any().item() is False, "phase output has NaNs"

    output = torch.polar(output[..., 0], output[..., 1])

    return output, hid_out


def model_eval_old(model, input_spec, configs, h0=None):
    abs_spectrum = input_spec.abs().to(input_spec.device)
    input_spec_ = torch.permute(torch.view_as_real(input_spec), dims=(0, 2, 3, 1))

    batch, frames, channels, frequency = input_spec_.shape
    abs_spectrum = torch.permute(abs_spectrum, dims=(0, 2, 1))
    abs_spectrum = torch.reshape(abs_spectrum, shape=(batch, frames, 1, frequency))
    if h0 is None:
        h0 = [
            [
                torch.zeros(
                    configs.dual_path_extension["parameters"]["num_layers"],
                    batch * configs.num_bands_out,
                    configs.dual_path_extension["parameters"]["inter_hidden_size"],
                    device=input_spec.device,
                )
                for _ in range(8)
            ]
            for _ in range(configs.dual_path_extension["num_modules"])
        ]

    output, abs_addon, hid_out = model(input_spec_, abs_spectrum, h0)

    output = torch.permute(output, dims=(0, 3, 1, 2))

    output = torch.view_as_complex(output.contiguous())
    out_abs, out_pha = output.abs(), output.angle()
    abs_addon = torch.permute(abs_addon, dims=(0, 3, 1, 2))

    out_abs = (out_abs + abs_addon[..., -1]) / 2

    result = torch.polar(out_abs, out_pha)

    return result, hid_out
