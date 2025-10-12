import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as taF

from models.ast_pcen import StatefulPCEN

from typing import Tuple

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
    # theta = torch.acos(cos_phase)
    # theta = torch.clamp(theta, min=-torch.pi, max=torch.pi)

    # cos_phase = torch.cos(theta)
    # cos_phase = torch.clamp(cos_phase, min=-1 + 1e-7, max=1 - 1e-7)

    # sin_phase = torch.sin(torch.acos(cos_phase))
    sin_phase = torch.sqrt(1 - cos_phase ** 2 + eps)
    sin_phase = torch.clamp(sin_phase, min=-1 + 1e-7, max=1 - 1e-7)
    # sin_phase = torch.sin(theta)

    # Now estimate the sign
    # q0 = x_features[:, 3:4, :]
    # q1 = x_features[:, 4:5, :]
    # print(q0.shape, q1.shape, torch.stack([q0, q1], dim=-2).shape)
    # tau = 1.
    # gamma = F.softmax(
    #     torch.stack([q0, q1], dim=-1) / tau, dim=-1
    # )
    # theta = torch.nn.functional.sigmoid(q0) * torch.pi
    # cos_phase = torch.cos(theta)  # torch.nn.functional.sigmoid(q0)
    # cos_phase = torch.clamp(cos_phase, min=-1 + 1e-7, max=1 - 1e-7)
    # sin_phase = torch.sqrt(1 - cos_phase ** 2 + eps)
    # sin_phase = torch.clamp(sin_phase, min=-1 + 1e-7, max=1 - 1e-7)
    #
    # tmp_1 = torch.isnan(mask_tf).sum().item()
    # tmp_2 = torch.isnan(cos_phase).sum().item()
    # tmp_3 = torch.isnan(sin_phase).sum().item()
    #
    # if tmp_1 > 0:
    #     print("Num of nuns in mask_tf: ", tmp_1)
    # if tmp_2 > 0:
    #     print("Num of nuns in cos_phase: ", tmp_2)
    # if tmp_3 > 0:
    #     print("Num of nuns in sin_phase: ", tmp_3)
    complex_mask = mask_tf * (cos_phase + 1j * sin_phase)
    complex_mask_residual = mask_tf_residual * (cos_phase + 1j * sin_phase)
    # print(complex_mask.shape)
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
            nn.SELU(inplace=True),
            # nn.ReLU(inplace=True),
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

def diff(x, axis):
    shape = x.shape
    begin_back = [0 for unused_s in range(len(shape))]
    begin_front = [0 for unused_s in range(len(shape))]
    begin_front[axis] = 1
    size = list(shape)
    size[axis] -= 1
    slice_front = x[begin_front[0]:begin_front[0] + size[0], begin_front[1]:begin_front[1] + size[1]]
    slice_back = x[begin_back[0]:begin_back[0] + size[0], begin_back[1]:begin_back[1] + size[1]]
    d = slice_front - slice_back
    return d


def unwrap(p, axis=-1):
    pi = torch.tensor(torch.acos(torch.zeros(1)).item() * 2)
    dd = diff(p, axis=axis)
    ddmod = torch.remainder(dd + pi, 2.0 * pi) - pi
    idx = torch.logical_and(torch.eq(ddmod, -pi), torch.greater(dd, 0))
    ddmod = torch.where(idx, torch.ones_like(ddmod) * pi, ddmod)
    ph_correct = ddmod - dd
    idx = torch.less(torch.abs(dd), pi)
    ddmod = torch.where(idx, torch.zeros_like(ddmod), dd)
    ph_cumsum = torch.cumsum(ph_correct, dim=axis)
    shape = torch.tensor(p.shape)
    shape[axis] = 1
    # ph_cumsum = torch.cat([torch.zeros(list(shape)), ph_cumsum], axis=axis)
    unwrapped = p + ph_cumsum
    return unwrapped.squeeze(0)


def demod_phase(phase):
    '''
    Calculates demodulated phase of real and imaginary
    Args:
        Phase (float32):        Phase of the clean signal
    Returns:
        real_demod (float32):   Demodulated phase of the real part of the clean signal
        imag_demod (float32):   Demodulated phase of imaginary the part of the clean signal
    '''

    demodulated_phase = unwrap(phase)

    # get real and imagniary parts of the demodulated phase
    real_demod = torch.sin(demodulated_phase)
    imag_demod = torch.cos(demodulated_phase)

    return real_demod, imag_demod


class DemodulatedPhase(nn.Module):
    def __init__(self, n_fft=512, hop_length=256, win_length=None, window='hann'):
        """
        Initialize demodulated phase computation.

        Args:
            n_fft: FFT size
            hop_length: Hop length between STFT frames
            win_length: Window length (defaults to n_fft)
            window: Window type ('hann', 'hamming', etc.)
        """
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft

        # Register window as buffer so it moves with the model
        window = torch.hann_window(self.win_length)
        self.register_buffer('window', window)

    def forward(self, stft: torch.Tensor) -> torch.Tensor:
        """
        Compute demodulated phase from time-domain signal.

        Args:
            x: Input signal [batch, samples]

        Returns:
            demod_phase: Demodulated phase [batch, freq, time]
        """
        # Get magnitude and phase
        mag = torch.abs(stft)
        phase = torch.angle(stft)

        # Compute instantaneous frequency (time derivative of phase)
        # Using finite differences between frames
        delta_phase = torch.diff(stft, dim=-1)

        # Wrap phase differences to [-pi, pi]
        delta_phase = torch.clamp(delta_phase, min=-torch.pi, max=torch.pi) # (delta_phase + torch.pi) % (2 * torch.pi) - torch.pi

        # Compute expected phase progression (linear component)
        # omega = 2*pi*f*T where T is hop_time
        freqs = torch.fft.fftfreq(self.n_fft)[:self.n_fft // 2 + 1]
        omega = 2 * torch.pi * freqs.to(stft.device) * (self.hop_length / 16_000)
        expected_delta = omega.unsqueeze(-1)  # [freq, 1]

        # Compute phase deviation (demodulated phase)
        # Pad delta_phase to match original time dimension
        delta_phase_padded = F.pad(delta_phase, (1, 0), mode='constant', value=0)
        phase_deviation = delta_phase_padded - expected_delta.unsqueeze(0)

        # Integrate deviations to get demodulated phase
        demod_phase = torch.cumsum(phase_deviation, dim=-1)

        return demod_phase

class TRUNet(nn.Module):
    def __init__(self, nfft=512, hop=128, sr=16_000, win='hann'):
        super(TRUNet, self).__init__()
        self.hop = 128
        self.sr = sr
        self.pcen = StatefulPCEN(smooth=0.025, trainable={"alpha": True, "delta": True, "root": True, "smooth": True})

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
        self.up6 = LastTrCNN(128, 1, 5, 2)

        self.demod_p = DemodulatedPhase(n_fft=nfft, hop_length=hop)

    @staticmethod
    def compute_demodulated_phase_torch(complex_spectrogram, hop_length, sample_rate):
        """
        Compute demodulated phase from a complex STFT spectrogram.

        Parameters:
        - complex_spectrogram: torch.Tensor of shape (B, F, T), complex dtype
        - hop_length: int, hop size used in the STFT
        - sample_rate: int, audio sample rate in Hz

        Returns:
        - demod_phase: torch.Tensor of shape (B, F, T), real-valued
        """
        # Extract wrapped phase
        phase = torch.angle(complex_spectrogram)  # (B, F, T)

        # Unwrap phase over time (dim=2)
        phase_diff = torch.diff(phase, dim=2)
        phase_diff = (phase_diff + torch.pi) % (2 * torch.pi) - torch.pi
        unwrapped_phase = torch.cat(
            [phase[:, :, :1], phase[:, :, :1] + torch.cumsum(phase_diff, dim=2)],
            dim=2
        )  # (B, F, T)

        # Time and frequency axes
        B, F, T = complex_spectrogram.shape
        times = torch.arange(T, device=phase.device).float() * hop_length / sample_rate  # (T,)
        freqs = torch.linspace(0, sample_rate / 2, F, device=phase.device).float()  # (F,)

        # Expected linear phase: 2π * freq * time
        expected_phase = 2 * torch.pi * freqs.unsqueeze(-1) * times.unsqueeze(0)  # (F, T)
        expected_phase = expected_phase.unsqueeze(0)  # (1, F, T) for batch broadcasting

        # Compute demodulated phase
        demod_phase = unwrapped_phase - expected_phase  # (B, F, T)

        return demod_phase

    def forward(self, x_spec, h0_f, h0_t):
        # print(torch.isnan(x).any(), ' x')

        # x_abs = x.abs()
        pcen_x = self.pcen(x_spec.abs().unsqueeze(1))[0]
        # logx = taF.amplitude_to_DB(x_abs.unsqueeze(1), amin=1e-4, top_db=80.0, multiplier=20.0,
        #                            db_multiplier=0.0)  # batch, channel, freq, time
        logx = torch.log1p(x_spec.abs().unsqueeze(1))

        demod_phase = TRUNet.compute_demodulated_phase_torch(x_spec, self.hop, self.sr)

        # x0 = torch.cat((pcen_x, logx, x_real.unsqueeze(1), x_imag.unsqueeze(1)), dim=1)
        x0 = torch.cat((pcen_x, logx, torch.cos(demod_phase).unsqueeze(1), torch.sin(demod_phase).unsqueeze(1)),
                       dim=1)
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

        mask_d = torch.abs(x16.reshape(bs, -1, time))

        # mask_d_complex, _ = calculate_PHM(x16[:, :5, :])
        # mask_n_complex, mask_n_residual_complex = calculate_PHM(x16[:, 5:, :])
        # mask_r_complex = mask_n_residual_complex - mask_d_complex
        # mask_d_complex = mask_d_complex.squeeze(1)
        #
        # mask_d_complex = mask_d_complex.reshape(bs, mask_d_complex.shape[1], time)
        # # mask_d_complex_abs = torch.abs(x16[:, 6:7, :].reshape(bs, mask_d_complex.shape[1], time))
        # # print(mask_d_complex_angle.shape, mask_d_complex_abs.shape)
        # mask_n_complex = mask_n_complex.reshape(bs, mask_d_complex.shape[1], time)
        # mask_r_complex = mask_r_complex.reshape(bs, mask_d_complex.shape[1], time)

        # return torch.polar(x_abs, x_ph) * mask_d_complex, mask_n_complex, mask_r_complex, h_f, h_t
        return mask_d, mask_d, mask_d, h_f, h_t


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
    wave_d, wave_n, wave_r, _, _ = TRU(x, h_f, h_t)
    print("output_shape:", wave_d.shape)# , wave_n.shape, wave_r.shape)
    # print(wave)
    # total params: 180336
    # input_shape: torch.Size([4, 257, 3])
    # output_shape: torch.Size([4, 257, 3])
