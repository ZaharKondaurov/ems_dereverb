import numpy as np
import torch

from termcolor import colored
from collections import defaultdict


def beautiful_int(i):
    i = str(i)
    return ".".join(reversed([i[max(j, 0):j+3] for j in range(len(i) - 3, -3, -3)]))


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

def model_eval(model, input_spec, device="cpu", hid_size=64):
    input_spec = input_spec.to(device)

    abs_spectrum = input_spec.abs()
    input_spec_ = torch.permute(torch.view_as_real(input_spec), dims=(0, 2, 3, 1))
    batch, frames, channels, frequency = input_spec_.shape
    abs_spectrum = torch.permute(abs_spectrum, dims=(0, 2, 1))
    abs_spectrum = torch.reshape(abs_spectrum, shape=(batch, frames, 1, frequency))
    h0 = [[torch.zeros(1, batch * hid_size, 16 // 8, device=input_spec.device) for _ in range(8)] for _ in range(3)]

    output, hid_out = model(input_spec_, abs_spectrum, h0)

    output = torch.permute(output, dims=(0, 3, 1, 2))
    output = torch.view_as_complex(output)

    return output, hid_out

def model_eval_fspen2x_ver3(model, input_spec, device="cpu", hid_size=64):
    input_spec = input_spec.to(device)

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
