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

    # sin_phase = torch.sin(torch.acos(cos_phase))
    sin_phase = torch.sqrt(1 - cos_phase ** 2)

    # Now estimate the sign
    q0 = x_features[:, 3:4, :]
    q1 = x_features[:, 4:5, :]
    # print(q0.shape, q1.shape, torch.stack([q0, q1], dim=-2).shape)
    # tau = 1.
    # gamma = F.softmax(
    #     torch.stack([q0, q1], dim=-1) / tau, dim=-1
    # )
    # # print(gamma.shape)
    # gamma_0 = gamma[..., 0]
    # gamma_1 = gamma[..., 1]
    # print(x_features[:, 3:5, :].shape)
    gamma = F.gumbel_softmax(x_features[..., 3:5, :], hard=True, dim=-2)
    sign = gamma[..., 0:1, :] - gamma[..., 1:2, :] # sign = torch.sign(gamma_0 - gamma_1)
    # print(gamma.shape, sign.shape, x_features.shape, cos_phase.shape)
    # print(sign.shape)
    # Finally, estimate the complex mask
    # print(torch.isnan(mask_tf).any())
    complex_mask = mask_tf * (cos_phase + sign * 1j * sin_phase)
    complex_mask_residual = mask_tf_residual * (cos_phase + sign * 1j * sin_phase)
    # print(complex_mask.shape)
    return complex_mask, complex_mask_residual


def compute_phm_mask(net_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the complex PHM (Phase-aware β-sigmoid Mask) for source separation.

    Args:
        net_output: Tensor of shape (T, 5, F), output from the neural network.
            - dim=1 index 0: z^{(k)}_{t,f}, input to sigmoid for source mask
            - dim=1 index 1: z^{(-k)}_{t,f}, input to sigmoid for rest mask
            - dim=1 index 2: φ_{t,f}, input to softplus for β_{t,f}
            - dim=1 index 3: logits for direction ξ_{t,f} ∈ {-1, 1}
            - dim=1 index 4: (optional, unused in this code)
        X_tf: Complex-valued input STFT of shape (T, F), dtype=torch.cfloat

    Returns:
        Estimated source Ŷ^{(k)}_{t,f}, shape (T, F), complex dtype.
    """

    # ----------------------
    # Step 1: Unpack components from the network output
    # ----------------------
    z_k = net_output[:, 0, :]  # z^{(k)}_{t,f}, for source mask
    z_neg_k = net_output[:, 1, :]  # z^{(-k)}_{t,f}, for complement mask
    phi = net_output[:, 2, :]  # φ_{t,f}, used to compute β_{t,f}
    direction_logits = net_output[:, 3, :]  # logits for ξ ∈ {-1, 1}

    # ----------------------
    # Step 2: Compute β_{t,f} using softplus to ensure β > 1
    # ----------------------
    # Softplus ensures positivity and allows the mask to extend beyond [0,1]
    # β_{t,f} = 1 + softplus(φ)
    beta_tf = 1 + F.softplus(phi)

    # ----------------------
    # Step 3: Compute magnitude masks using the β-sigmoid function
    # ----------------------
    # Magnitude masks are flexible sigmoid-based and scaled by β
    sig_k = torch.sigmoid(z_k)  # σ(z^{(k)}_{t,f})
    sig_neg_k = torch.sigmoid(z_neg_k)  # σ(z^{(-k)}_{t,f})
    abs_M_k = beta_tf * sig_k  # |M^{(k)}_{t,f}|
    abs_M_neg_k = beta_tf * sig_neg_k  # |M^{(-k)}_{t,f}|

    # ----------------------
    # Step 4: Enforce triangle inequality on |M^{(k)}| and |M^{(-k)}|
    # ----------------------
    # To ensure that the complex vectors can form a triangle:
    # | |M^{(k)}| - |M^{(-k)}| | ≤ 1 and |M^{(k)}| + |M^{(-k)}| ≥ 1
    # First inequality is enforced by clipping β using the inverse difference
    eps = 1e-8  # To avoid division by zero
    diff = (sig_k - sig_neg_k).abs().clamp(min=eps)
    beta_upper_bound = 1.0 / diff
    beta_tf_clipped = torch.minimum(beta_tf, beta_upper_bound)

    # Recalculate masks using the clipped β
    abs_M_k = beta_tf_clipped * sig_k
    abs_M_neg_k = beta_tf_clipped * sig_neg_k

    # ----------------------
    # Step 5: Compute cos(Δθ_{t,f}^{(k)}) using the cosine law
    # ----------------------
    # We use the triangle formed by |X|, |Y^{(k)}| = |M^{(k)}||X|, |Y^{(-k)}| = |M^{(-k)}||X|
    # Using cosine law: cos(Δθ) = (a^2 + c^2 - b^2) / 2ac
    # But everything is in terms of masks, so we use:
    # cos(Δθ) = (1 + |M^{(k)}|^2 - |M^{(-k)}|^2) / (2 |M^{(k)}|)
    numerator = 1 + abs_M_k ** 2 - abs_M_neg_k ** 2
    denominator = 2 * abs_M_k.clamp(min=eps)  # Avoid div by zero
    cos_delta_theta = (numerator / denominator).clamp(-1 + eps, 1 - eps)

    # ----------------------
    # Step 6: Estimate sign direction ξ_{t,f} ∈ {-1, 1} using Gumbel-softmax
    # ----------------------
    # Used to resolve the sign ambiguity of sin(Δθ)
    direction = F.gumbel_softmax(direction_logits, tau=1.0, hard=True)
    # print(direction)
    # Convert one-hot to ±1: direction[:,1] → 1, direction[:,0] → -1
    # xi_tf = (direction[:, 1] * 2 - 1)[..., None]    # Что-то странное
    direction = direction * 2 - 1

    # ----------------------
    # Step 7: Construct the complex phase mask: e^{jθ} = cos(Δθ) + j ξ sin(Δθ)
    # ----------------------
    sin_delta_theta = torch.sqrt(1 - cos_delta_theta ** 2)
    phase_real = cos_delta_theta
    # print(xi_tf.shape, sin_delta_theta.shape, direction.shape)
    phase_imag = direction * sin_delta_theta
    phase_mask = torch.complex(phase_real, phase_imag)

    # ----------------------
    # Step 8: Form final complex-valued mask M = |M| * e^{jθ}
    # ----------------------
    complex_mask = abs_M_k * phase_mask
    complex_mask_res = abs_M_neg_k * phase_mask
    # ----------------------
    # Step 9: Apply the complex mask to the input mixture to estimate the source
    # ----------------------
    # Y_hat_k = complex_mask * X_tf  # Final estimated complex source

    return complex_mask, complex_mask_res


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

        self.demod_p = DemodulatedPhase(n_fft=nfft, hop_length=hop)

    def forward(self, x_abs, x_real, x_imag, h0_f, h0_t):
        # print(torch.isnan(x).any(), ' x')

        # x_abs = x.abs()
        pcen_x = self.pcen(x_abs.unsqueeze(1))[0]
        logx = taF.amplitude_to_DB(x_abs.unsqueeze(1), amin=1e-4, top_db=80.0, multiplier=20.0,
                                   db_multiplier=0.0)  # batch, channel, freq, time
        # real = x.real
        # imag = x.imag

        # angle = x_spec.angle()
        # demod = self.demod_p(x_spec)
        # real_demod, imag_demod = demod_phase(angle)

        x0 = torch.cat((pcen_x, logx, x_real.unsqueeze(1), x_imag.unsqueeze(1)), dim=1)
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
        # mask_d = x16[:, 0, ...].reshape(x_abs.shape[0], 257, -1) * x_abs
        # mask_n = x16[:, 1, ...].reshape(x_abs.shape[0], 257, -1) * x_abs
        # mask_r = x16[:, 2, ...].reshape(x_abs.shape[0], 257, -1) * x_abs
        # x16 = x16[:, 0, ...].reshape(x_abs.shape[0], 257, -1)
        # print(x16.shape, x_abs.shape)
        # mask_d = x16 * x_abs

        # mask_d, _ = calculate_PHM(x16[:, :5, :])
        # mask_n, mask_n_residual = calculate_PHM(x16[:, 5:, :])
        # mask_d, _ = calculate_PHM(x16[:, :5, :])
        # mask_n, mask_n_residual = calculate_PHM(x16[:, 5:, :])
        # mask_r = mask_n_residual - mask_d
        mask_d = x16[:, 0, ...].reshape(x_abs.shape[0], 257, -1)
        mask_n = x16[:, 1, ...].reshape(x_abs.shape[0], 257, -1)
        mask_r = x16[:, 2, ...].reshape(x_abs.shape[0], 257, -1)
        # print(mask_d.shape)
        # mask_d = mask_d.reshape(bs, mask_d.shape[1], mask_d.shape[2], time)
        # mask_n = mask_n.reshape(bs, mask_n.shape[1], mask_n.shape[2], time)
        # mask_r = mask_r.reshape(bs, mask_r.shape[1], mask_r.shape[2], time)
        # mask_d = mask_d.reshape(bs, mask_d.shape[1], time)
        # mask_n = mask_n.reshape(bs, mask_n.shape[1], time)
        # mask_r = mask_r.reshape(bs, mask_r.shape[1], time)
        # return mask_d.squeeze(1), mask_n.squeeze(1), mask_r.squeeze(1), h_f, h_t
        return mask_d * x_abs, mask_n * x_abs, mask_r * x_abs, h_f, h_t


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
