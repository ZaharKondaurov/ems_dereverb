import torch
from torch import nn
import torch.nn.functional as F

from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from models.fspen import FullSubPathExtension, DiscriminatorModel
from src.fspen_configs import TrainConfig
from src.utils import use_pcs


def _compute_mr(Y: torch.Tensor, Y_abs: torch.Tensor, S: torch.Tensor, S_abs: torch.Tensor) -> torch.Tensor:
    x1 = F.mse_loss(Y_abs, S_abs)
    x2 = F.mse_loss(torch.abs(Y_abs * (Y / (torch.abs(Y) + 1e-9))), torch.abs(S_abs * (S / (torch.abs(S) + 1e-9))))
    # print(f"Abs: {x1:.2f}, Pha: {x2:.2f}")
    return x1 + x2

def anti_wrapping_function(x):
    return torch.abs(x - torch.round(x / (2 * torch.pi)) * 2 * torch.pi)

def loss_MR(input: torch.Tensor, target: torch.Tensor, gamma: float = 1.0, nffts: list = None, hop_fr: float = 0.25, low_freq_ratio: float = 0.25, pcs: bool = False) -> torch.Tensor:
    if nffts is None:
        nffts = [1024, 512, 256]
    # print(input.isnan().any(), target.isnan().any(), ' inputs of MR loss')
    loss = torch.zeros((), device=input.device, dtype=input.dtype)
    for nfft in nffts:
        Y = torch.stft(
            input,
            n_fft=nfft,
            hop_length=int(nfft * hop_fr),
            window=torch.hann_window(nfft, device=input.device),
            normalized=True,
            return_complex=True,
        )
        S = torch.stft(
            target,
            n_fft=nfft,
            hop_length=int(nfft * hop_fr),
            window=torch.hann_window(nfft, device=target.device),
            normalized=True,
            return_complex=True,
        )

        if not pcs:
            Y = use_pcs(Y, nfft)
            
        Y_abs = Y.abs()
        S_abs = S.abs()
        # Try loss for angle
        # Y_angle = Y.angle()
        # S_angle = S.angle()
        if (gamma != 1) and (not pcs):
            Y_abs = Y_abs.clamp_min(1e-12).pow(gamma)
            S_abs = S_abs.clamp_min(1e-12).pow(gamma)
            # Y_angle = Y_angle.clamp_min(1e-12).pow(gamma)
            # S_angle = S_angle.clamp_min(1e-12).pow(gamma)
        # print(Y_abs.shape, nfft)
        low_sub_band_0 = int((nfft // 2 + 1) * 0.1)
        low_sub_band_1 = int((nfft // 2 + 1) * 0.25)
        low_sub_band_2 = int((nfft // 2 + 1) * 0.5)
        loss += _compute_mr(Y, Y_abs, S, S_abs)
                #  _compute_mr(Y[..., :low_sub_band_0, :], Y_abs[..., :low_sub_band_0, :], S[..., :low_sub_band_0, :], S_abs[..., :low_sub_band_0, :]) +
                #  _compute_mr(Y[..., :low_sub_band_1, :], Y_abs[..., :low_sub_band_1, :], S[..., :low_sub_band_1, :], S_abs[..., :low_sub_band_1, :]) +
                #  _compute_mr(Y[..., :low_sub_band_2, :], Y_abs[..., :low_sub_band_2, :], S[..., :low_sub_band_2, :], S_abs[..., :low_sub_band_2, :]))
    #     print(loss.shape, loss, f' !!!!!!!!!lossMR for {nfft}!!!!!!!!!!!!')
    # print(loss.shape, loss, ' !!!!!!!!!lossMR!!!!!!!!!!!!')
    return loss / len(nffts) # + loss_MR_w(input, target) * 0.3


def loss_MR_PCS(input: torch.Tensor, target: torch.Tensor, nffts: list = None, hop_fr: float = 0.25) -> torch.Tensor:
    if nffts is None:
        nffts = [1024, 512, 256]
    loss = torch.zeros((), device=input.device, dtype=input.dtype)
    for nfft in nffts:
        Y = torch.stft(
            input,
            n_fft=nfft,
            hop_length=int(nfft * hop_fr),
            window=torch.hann_window(nfft, device=input.device),
            normalized=True,
            return_complex=True,
        )
        S = torch.stft(
            target,
            n_fft=nfft,
            hop_length=int(nfft * hop_fr),
            window=torch.hann_window(nfft, device=target.device),
            normalized=True,
            return_complex=True,
        )

        Y = use_pcs(Y, n_fft=nfft)

        Y_abs = Y.abs()
        S_abs = S.abs()
        # if gamma != 1:
        #     Y_abs = Y_abs.clamp_min(1e-12).pow(gamma)
        #     S_abs = S_abs.clamp_min(1e-12).pow(gamma)

        loss += _compute_mr(Y, Y_abs, S, S_abs)
    return loss / len(nffts)

def loss_pcm(predicted: torch.Tensor, target: torch.Tensor, input: torch.Tensor, nffts: list = None, hop_fr: float = 0.25):
    if nffts is None:
        nffts = [1024, 512, 256]
    loss = torch.zeros((), device=input.device, dtype=input.dtype)
    for nfft in nffts:
        X = torch.view_as_real(torch.stft(
            predicted,
            n_fft=nfft,
            hop_length=int(nfft * hop_fr),
            window=torch.hann_window(nfft, device=input.device),
            normalized=True,
            return_complex=True,
        ))
        Y = torch.view_as_real(torch.stft(
            input,
            n_fft=nfft,
            hop_length=int(nfft * hop_fr),
            window=torch.hann_window(nfft, device=input.device),
            normalized=True,
            return_complex=True,
        ))
        X_target = torch.view_as_real(torch.stft(
            target,
            n_fft=nfft,
            hop_length=int(nfft * hop_fr),
            window=torch.hann_window(nfft, device=target.device),
            normalized=True,
            return_complex=True,
        ))

        loss_1 = F.l1_loss(X.sum(-1).abs(), X_target.sum(-1).abs())
        loss_2 = F.l1_loss((Y - X).sum(-1).abs(), (Y - X_target).sum(-1).abs())

        loss += loss_1 + loss_2

    return loss / len(nffts)

def loss_ri_mag(predicted: torch.Tensor, target: torch.Tensor, nffts: list = None, hop_fr: float = 0.25):
    if nffts is None:
        nffts = [1024, 512, 256]
    loss = torch.zeros((), device=predicted.device, dtype=predicted.dtype)
    for nfft in nffts:
        X = torch.view_as_real(torch.stft(
            predicted,
            n_fft=nfft,
            hop_length=int(nfft * hop_fr),
            window=torch.hann_window(nfft, device=predicted.device),
            normalized=True,
            return_complex=True,
        ))
        Y = torch.view_as_real(torch.stft(
            target,
            n_fft=nfft,
            hop_length=int(nfft * hop_fr),
            window=torch.hann_window(nfft, device=target.device),
            normalized=True,
            return_complex=True,
        ))

        loss_1 = F.l1_loss(X.sum(-1).abs(), Y.sum(-1).abs())
        loss_2 = F.l1_loss(torch.view_as_complex(X).abs(), torch.view_as_complex(Y).abs())

        loss += loss_1 + loss_2

    return loss / len(nffts)

def phase_losses(phase_r, phase_g):
    """
    Calculate phase losses including in-phase loss, gradient delay loss, 
    and integrated absolute frequency loss between reference and generated phases.
    
    Args:
        phase_r (torch.Tensor): Reference phase tensor of shape (batch, freq, time).
        phase_g (torch.Tensor): Generated phase tensor of shape (batch, freq, time).
        h (object): Configuration object containing parameters like n_fft.
    
    Returns:
        tuple: Tuple containing in-phase loss, gradient delay loss, and integrated absolute frequency loss.
    """
    dim_freq = phase_r.size(-2)  # Calculate frequency dimension
    dim_time = phase_r.size(-1)  # Calculate time dimension
    
    # Construct gradient delay matrix
    gd_matrix = (torch.triu(torch.ones(dim_freq, dim_freq), diagonal=1) - 
                 torch.triu(torch.ones(dim_freq, dim_freq), diagonal=2) - 
                 torch.eye(dim_freq)).to(phase_g.device)
    
    # Apply gradient delay matrix to reference and generated phases
    gd_r = torch.matmul(phase_r.permute(0, 2, 1), gd_matrix)
    gd_g = torch.matmul(phase_g.permute(0, 2, 1), gd_matrix)
    
    # Construct integrated absolute frequency matrix
    iaf_matrix = (torch.triu(torch.ones(dim_time, dim_time), diagonal=1) - 
                  torch.triu(torch.ones(dim_time, dim_time), diagonal=2) - 
                  torch.eye(dim_time)).to(phase_g.device)
    
    # Apply integrated absolute frequency matrix to reference and generated phases
    iaf_r = torch.matmul(phase_r, iaf_matrix)
    iaf_g = torch.matmul(phase_g, iaf_matrix)
    
    # Calculate losses
    ip_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))
    gd_loss = torch.mean(anti_wrapping_function(gd_r - gd_g))
    iaf_loss = torch.mean(anti_wrapping_function(iaf_r - iaf_g))
    
    return ip_loss, gd_loss, iaf_loss

def loss_MR_w(input: torch.Tensor, target: torch.Tensor, lens: list = None):
    if lens is None:
        lens = [4064, 2032,] #1016, 508]
    # print(input.isnan().any(), target.isnan().any(), ' inputs of MRw loss')
    loss = torch.zeros((), device=input.device, dtype=input.dtype)
    for seg in lens:
        input_chunks = torch.split(input, seg, dim=1)
        target_chunks = torch.split(target, seg, dim=1)
        assert len(input_chunks) == len(target_chunks), f"{len(input_chunks)} != {len(target_chunks)}"
        loss_interm = torch.zeros((input.shape[0],), device=input.device, dtype=input.dtype)
        for in_ch, tg_ch in zip(input_chunks, target_chunks):
            cossim = -F.cosine_similarity(in_ch, tg_ch) + 1
            loss_interm += cossim
        # print(loss_interm.shape, ' cossim loss')
        loss += torch.sum(loss_interm, dim=0) / loss_interm.shape[0] / len(input_chunks)
        # print(loss, f' !!!!!!!!!lossMRw for {seg}!!!!!!!!!!!!')
    # print(loss.shape, loss,  ' !!!!!!!!!lossMRw!!!!!!!!!!!!')
    return loss / len(lens)


def loss_tot(input_signal: torch.Tensor, target: torch.Tensor,
             noise: torch.Tensor = None, target_noise: torch.Tensor = None,
             rir: torch.Tensor = None, target_rir: torch.Tensor = None,
             gamma: float = 0.3, nffts: list = None, hop_fr: float = 0.75, lens: list = None):

    loss_mr_w = loss_MR_w(input_signal, target, lens)
    loss_mr = loss_MR(input_signal, target, gamma, nffts, hop_fr)

    if noise is not None and target_noise is not None:
        loss_mr_w += loss_MR_w(noise, target_noise, lens)
        loss_mr += loss_MR(noise, target_noise, gamma, nffts, hop_fr)
    if rir is not None and target_rir is not None:
        loss_mr_w += loss_MR_w(rir, target_rir, lens)
        loss_mr += loss_MR(rir, target_rir, gamma, nffts, hop_fr)

    # loss_mr = (loss_MR(input_signal, target, gamma, nffts, hop_fr) + loss_MR(noise, target_noise, gamma, nffts, hop_fr)
    #            + loss_MR(rir, target_rir, gamma, nffts, hop_fr))
    return loss_mr_w + 2 * loss_mr


class MagnitudeConsistencyLoss(nn.Module):

    def __init__(self):
        super(MagnitudeConsistencyLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, gt, x):
        return self.mse(gt, x)


class DiscriminatorLoss(nn.Module):

    def __init__(self, sr=16_000, mode="wb", device="cpu"):
        super(DiscriminatorLoss, self).__init__()
        self.mse_1 = nn.MSELoss()
        self.mse_2 = nn.MSELoss()
        self.pesq = PerceptualEvaluationSpeechQuality(sr, mode)
        self.device = device

    def forward(self, d_m, d_cons, gt_wave, x_wave):
        # pesq_ = (self.pesq(x_wave, gt_wave) + 0.5) / 5.

        return (self.mse_1(d_m, torch.zeros_like(d_m, device=self.device))
                + self.mse_2(d_cons, torch.ones_like(d_cons, device=self.device)))


class MetricLoss(nn.Module):
    def __init__(self, device="cpu"):
        super(MetricLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.device = device

    def forward(self, d_cons):
        return self.mse(d_cons, torch.ones_like(d_cons, device=self.device))


class GeneratorLoss(nn.Module):
    def __init__(self, l1=1.0, l2=0., device="cpu"):
        super(GeneratorLoss, self).__init__()
        self.l1 = l1
        self.l2 = l2
        self.mag_loss = MagnitudeConsistencyLoss()
        self.metric = MetricLoss(device)

    def forward(self, gt, x, d_m):
        return self.l1 * loss_tot(x, gt) + self.l2 * self.metric(d_m)


if __name__ == '__main__':
    d_loss = DiscriminatorLoss()
    g_loss = GeneratorLoss(0.5, 0.5)

    x_ = torch.randn([2, 126, 2, 257])
    x_abs = torch.randn([2, 126, 1, 257])
    h0 = [[torch.randn(1, 2 * 32, 16 // 8) for _ in range(8)]
          for _ in range(3)]

    configs = TrainConfig()

    trunet = FullSubPathExtension(configs=configs)

    model = FullSubPathExtension(configs=configs)
    output, _ = model(x_, x_abs, h0)
    print(output.shape)
    discr = DiscriminatorModel(c_in=2)

    discr_output = discr(output)
    print(discr_output.shape)
    output = output.reshape(2 * 2, output.shape[1], output.shape[3])
    g_out = g_loss(output, output, discr_output)
    print(g_out)
    d_out = d_loss(discr_output, discr_output, torch.ones((1, 16_000)), torch.ones((1, 16_000)))
    print(d_out)
