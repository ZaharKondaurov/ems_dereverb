import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as taF

from models.ast_pcen import StatefulPCEN


def calculate_PHM(
        x_features):  # так или иначе проблема где-то тут, мб вынести это за модель и считать по отдельным бинам
    eps = 1e-8
    z_tf = x_features[:, 0:1, :]
    z_tf_residual = x_features[:, 1:2, :]
    # print(torch.isnan(z_tf).any(), torch.isnan(z_tf_residual).any(), ' z_tf, residual')

    # Extract phi
    phi = x_features[:, 2:3, :]

    # Estimate beta (due to softplus it will be one or greater)
    beta = 1.0 + F.softplus(phi)

    # print(torch.isnan(beta).any(), torch.isnan(phi).any(), ' phi beta')

    # Estimate sigmod of target and residual
    sigmoid_tf = F.sigmoid(z_tf - z_tf_residual)
    sigmoid_tf_residual = 1 - sigmoid_tf

    # print(torch.isnan(sigmoid_tf).any(), torch.isnan(sigmoid_tf_residual).any(), ' sigmoid_tf sigmoid_tf_residual')

    # Estimate upper bound for beta
    beta_upper_bound = 1.0 / (torch.abs(sigmoid_tf - sigmoid_tf_residual) + eps)

    # print(torch.isnan(beta_upper_bound).any(), ' beta_upper_bound')

    # Because of the absolute value in the denominator, the same upper bound
    # can be applied to both betas
    beta = torch.clip(beta, max=beta_upper_bound)

    # Compute both target and residual mks using eq. (1)
    # print(torch.isnan(beta).any(), torch.isnan(sigmoid_tf).any(), ' beta, sigmoid_tf')
    mask_tf = beta * sigmoid_tf
    mask_tf_residual = beta * sigmoid_tf_residual

    # Now that we have both masks, let's compute the triangle cosine law

    cos_phase = (
            (1.0 + mask_tf.square() - mask_tf_residual.square())
            / (2.0 * mask_tf + eps))

    cos_phase = torch.clamp(cos_phase, min=-1 + 1e-7, max=1 - 1e-7)

    sin_phase = torch.sin(torch.acos(cos_phase))

    # Now estimate the sign
    q0 = x_features[:, 3:4, :]
    q1 = x_features[:, 4:5, :]
    # print(q0.shape, q1.shape, torch.stack([q0, q1], dim=-2).shape)
    tau = 0.5
    gamma = F.softmax(
        torch.stack([q0, q1], dim=-1) / tau, dim=-1
    )
    # print(gamma.shape)
    gamma_0 = gamma[..., 0]
    gamma_1 = gamma[..., 1]

    sign = torch.sign(gamma_0 - gamma_1)

    # Finally, estimate the complex mask
    # print(torch.isnan(mask_tf).any())
    complex_mask = mask_tf * (cos_phase + sign * 1j * sin_phase)
    complex_mask_residual = mask_tf_residual * (cos_phase + sign * 1j * sin_phase)

    return complex_mask, complex_mask_residual


class StandardConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(StandardConv1d, self).__init__()
        self.StandardConv1d = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=stride // 2),     # Может, kernel_size?
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.StandardConv1d(x)


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DepthwiseSeparableConv1d, self).__init__()
        self.DepthwiseSeparableConv1d = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,
                      out_channels=in_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=kernel_size // 2,
                      groups=in_channels),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.DepthwiseSeparableConv1d(x)


class GRUBlock(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels, bidirectional, batch_first):
        super(GRUBlock, self).__init__()
        self.GRU = nn.GRU(in_channels, hidden_size, batch_first=batch_first, bidirectional=bidirectional)

        self.conv = nn.Sequential(
            nn.Conv1d(hidden_size * (2 if bidirectional == True else 1), out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x, h0):
        # print(x.shape, '!!!!!!!')
        output, h = self.GRU(x, h0)
        output = output.transpose(1, 2)
        output = self.conv(output)
        return output, h


class FirstTrCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(FirstTrCNN, self).__init__()
        self.FirstTrCNN = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=stride // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.FirstTrCNN(x)


class TrCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(TrCNN, self).__init__()
        self.TrCNN = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=stride // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diffY // 2, diffY - diffY // 2, 0, 0])
        x = torch.cat((x1, x2), 1)
        output = self.TrCNN(x)
        return output


class LastTrCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(LastTrCNN, self).__init__()
        self.LastTrCNN = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=stride // 2))

    def forward(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diffY // 2, diffY - diffY // 2, 0, 0])
        x = torch.cat((x1, x2), 1)
        output = self.LastTrCNN(x)
        return output


class TRUNet(nn.Module):
    def __init__(self, nfft=512, hop=128, win='hann'):
        super(TRUNet, self).__init__()
        self.pcen = StatefulPCEN(smooth=0.025, trainable={"alpha": True, "delta": False, "root": True, "smooth": False})

        # self.down1 = StandardConv1d(4, 16, 4, 2)
        # self.down2 = DepthwiseSeparableConv1d(16, 32, 3, 1)
        # self.down3 = DepthwiseSeparableConv1d(32, 32, 5, 2)
        # self.down4 = DepthwiseSeparableConv1d(32, 32, 3, 1)
        # self.down5 = DepthwiseSeparableConv1d(32, 32, 5, 2)
        # self.down6 = DepthwiseSeparableConv1d(32, 64, 3, 2)
        # self.FGRU = GRUBlock(64, 64, 64, bidirectional=True, batch_first=True)
        # self.TGRU = GRUBlock(64, 128, 64, bidirectional=False, batch_first=False)
        # self.up1 = FirstTrCNN(64, 32, 3, 2)
        # self.up2 = TrCNN(64, 32, 5, 2)
        # self.up3 = TrCNN(64, 32, 3, 1)
        # self.up4 = TrCNN(64, 32, 5, 2)
        # self.up5 = TrCNN(64, 32, 3, 1)
        # self.up6 = LastTrCNN(48, 10, 5, 2)  # 5 kernel with dc bin, 4 kernel for cut dc

        self.down1 = StandardConv1d(4, 64, 5, 2)
        self.down2 = DepthwiseSeparableConv1d(64, 128, 3, 1)
        self.down3 = DepthwiseSeparableConv1d(128, 128, 5, 2)
        self.down4 = DepthwiseSeparableConv1d(128, 128, 3, 1)
        self.down5 = DepthwiseSeparableConv1d(128, 128, 5, 2)
        self.down6 = DepthwiseSeparableConv1d(128, 128, 3, 2)
        self.FGRU = GRUBlock(128, 64, 64, bidirectional=True, batch_first=True)
        self.TGRU = GRUBlock(64, 128, 64, bidirectional=False, batch_first=False)
        self.up1 = FirstTrCNN(64, 64, 3, 2)
        self.up2 = TrCNN(192, 64, 5, 2)
        self.up3 = TrCNN(192, 64, 3, 1)
        self.up4 = TrCNN(192, 64, 5, 2)
        self.up5 = TrCNN(192, 64, 3, 1)
        self.up6 = LastTrCNN(128, 10, 5, 2)

    def forward(self, x_abs, real, imag, h0_f, h0_t):
        # print(torch.isnan(x).any(), ' x')

        # x_abs = x.abs()
        pcen_x = self.pcen(x_abs.unsqueeze(1))[0]
        logx = taF.amplitude_to_DB(x_abs.unsqueeze(1), amin=1e-4, top_db=80.0, multiplier=20.0,
                                   db_multiplier=0.0)  # batch, channel, freq, time
        # real = x.real
        # imag = x.imag
        x0 = torch.cat((pcen_x, logx, real.unsqueeze(1), imag.unsqueeze(1)), dim=1)
        # print(x0.shape, ' x0')
        bs = x0.shape[0]
        time = x0.shape[-1]
        x0 = x0.view(bs * time, x0.shape[1], x0.shape[2])
        # print(x0.shape, ' x0')
        x1 = self.down1(x0)
        # print(x1.shape, ' x1')
        x2 = self.down2(x1)
        # print(x2.shape, ' x2')
        x3 = self.down3(x2)
        # print(x3.shape, ' x3')
        x4 = self.down4(x3)
        # print(x4.shape, ' x4')
        x5 = self.down5(x4)
        # print(x5.shape, ' x5')
        x6 = self.down6(x5)
        # print(x6.shape, ' x6 after all convs')
        x7 = x6.transpose(1, 2)
        # print(x7.shape, ' x7', h0_f.shape, ' h0_f')
        x8, h_f = self.FGRU(x7, h0_f)
        # print(x8.shape, ' x8 after FGRU')
        x9 = x8.transpose(1, 2)
        # print(x9.shape, ' x9', h0_t.shape, ' h0_t')
        x10, h_t = self.TGRU(x9, h0_t)
        # print(x10.shape, ' x10 after TGRU')
        x11 = self.up1(x10)
        # print(x11.shape, x5.shape)
        x12 = self.up2(x11, x5)
        x13 = self.up3(x12, x4)
        x14 = self.up4(x13, x3)
        x15 = self.up5(x14, x2)
        x16 = self.up6(x15, x1)
        x16 = x16[:, 0, ...].reshape(x_abs.shape[0], 257, -1)
        # print(x16.shape, x_abs.shape)
        mask_d = x16 * x_abs

        # mask_d, _ = calculate_PHM(x16[:, :5, :])
        # mask_n, mask_n_residual = calculate_PHM(x16[:, 5:, :])
        # mask_r = mask_n_residual - mask_d
        # # mask = x16[:, 0:1, :]
        # mask_d = mask_d.reshape(bs, mask_d.shape[1], mask_d.shape[2], time)
        # mask_n = mask_n.reshape(bs, mask_n.shape[1], mask_n.shape[2], time)
        # mask_r = mask_r.reshape(bs, mask_r.shape[1], mask_r.shape[2], time)

        return mask_d.squeeze(1), None, None, h_f, h_t# .squeeze(1)# , # mask_n.squeeze(1), mask_r.squeeze(1), h_f, h_t


if __name__ == '__main__':
    TRU = TRUNet()
    total_params = sum(p.numel() for p in TRU.parameters())
    print("total params:", total_params)
    h_f = torch.randn(2, 12, 64)
    h_t = torch.randn(1, 16, 128)
    _audio = torch.randn((4, 1024))
    x = torch.stft(
        _audio,
        n_fft=512,
        hop_length=256,
        # onesided=True,
        window=torch.hann_window(512),
        return_complex=True,
        normalized=True,
        center=False
    )

    print("input_shape:", x.shape)
    wave_d, wave_n, wave_r, _, _ = TRU(x.abs(), x.real, x.imag, h_f, h_t)
    print("output_shape:", wave_d.shape)# , wave_n.shape, wave_r.shape)
    # print(wave)
    # total params: 180336
    # input_shape: torch.Size([4, 257, 3])
    # output_shape: torch.Size([4, 257, 3])
