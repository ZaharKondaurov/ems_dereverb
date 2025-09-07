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
        if padding is None:
            self.left_pad = kernel_size - 1
            self.right_pad = stride - 1
        else:
            if stride != 1 and padding != kernel_size - 1:
                raise ValueError("No striding allowed for non-symmetric convolutions!")
            if isinstance(padding, int):
                self.left_padding = padding
                self.right_padding = padding
            elif isinstance(padding, list) and len(padding) == 2 and padding[0] + padding[1] == kernel_size - 1:
                self.left_padding = padding[0]
                self.right_padding = padding[1]
            else:
                raise ValueError(f"Invalid padding param: {padding}!")

        self.conv = nn.Conv1d(
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
        x = torch.nn.functional.pad(x, (self.left_padding, self.right_padding))
        return self.conv(x)


class CausalConvTranspose1d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0,
                 dilation: int = 1, groups: int = 1, bias: bool = True):
        super(CausalConvTranspose1d, self).__init__()
        self.trim = padding * 2 # max(padding + (kernel_size - 1) * dilation - stride - 1, 0)
        # self.trim -= self.trim % 2
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride,
                                       padding=0, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        # print(x.shape, self.trim)
        return x[..., :-self.trim]


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
