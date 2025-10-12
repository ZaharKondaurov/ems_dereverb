import torch
from torch import nn, Tensor


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True
    ):
        super(CausalConv1d, self).__init__()
        self.left_pad = kernel_size * dilation // 2 # abs(padding - (kernel_size - 1) * dilation)
        self.right_pad = self.left_pad // stride # self.right_pad = abs(padding - (stride - 1))
        self.padding = padding
        self.dilation = dilation
        self.stride = stride
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        l_out = (x.shape[-1] + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1

        x = torch.nn.functional.pad(x, (self.left_pad, 0))
        x = self.conv(x)[..., :l_out]
        # print(x.shape)
        return x

# torch.Size([569, 4, 128])
# torch.Size([569, 16, 64])
# torch.Size([569, 64, 32])
# torch.Size([569, 64, 32])
# torch.Size([569, 64, 8])
# torch.Size([569, 64, 6])
# torch.Size([569, 64, 6])
# torch.Size([569, 64, 6])
# torch.Size([569, 64, 6])
# torch.Size([569, 32, 32])
# torch.Size([569, 64, 32])
# torch.Size([569, 64, 32])

class CausalConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True
    ):
        super(CausalConvTranspose1d, self).__init__()
        # self.left_pad = kernel_size - 1
        self.right_pad = padding + kernel_size // 2 - 1
        self.conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)[..., :-self.right_pad]
        # print(x.shape)
        return x

# torch.Size([569, 16, 64])
# torch.Size([569, 4, 128])
# torch.Size([569, 2, 256])

# class CausalConv1d(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         kernel_size: int,
#         stride: int = 1,
#         padding: int = 0,
#         dilation: int = 1,
#         groups: int = 1,
#         bias: bool = True
#     ):
#         super(CausalConv1d, self).__init__()
#         self.pad = padding * 2 # max(padding + (kernel_size - 1) * dilation - stride - 1, 0)
#         self.conv = nn.Conv1d(
#             in_channels,
#             out_channels,
#             kernel_size,
#             stride=stride,
#             padding=0,
#             dilation=dilation,
#             groups=groups,
#             bias=bias
#         )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = torch.nn.functional.pad(x, (self.pad, 0))
#         return self.conv(x)
#
#
# class CausalConvTranspose1d(nn.Module):
#
#     def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0,
#                  dilation: int = 1, groups: int = 1, bias: bool = True):
#         super(CausalConvTranspose1d, self).__init__()
#         self.trim = padding * 2 # max(padding + (kernel_size - 1) * dilation - stride - 1, 0)
#         # self.trim -= self.trim % 2
#         self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride,
#                                        padding=0, dilation=dilation, groups=groups, bias=bias)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.conv(x)
#         # print(x.shape, self.trim)
#         return x[..., :-self.trim]
