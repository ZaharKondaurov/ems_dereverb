import torch
from torch import nn
import torch.nn.functional as F

from models.fspen import FullSubPathExtension
from src.fspen_configs import TrainConfig
from src.utils import vorbis_window


def _compute_mr(Y: torch.Tensor, Y_abs: torch.Tensor, S: torch.Tensor, S_abs: torch.Tensor) -> torch.Tensor:
    x1 = F.mse_loss(Y_abs, S_abs)
    x2 = F.mse_loss(torch.abs(Y_abs * (Y / (torch.abs(Y) + 1e-9))), torch.abs(S_abs * (S / (torch.abs(S) + 1e-9))))
    return x1 + x2

def anti_wrapping_function(x):
    return torch.abs(x - torch.round(x / (2 * torch.pi)) * 2 * torch.pi)

def loss_clipping_penalty(input: torch.Tensor, tau: float = 0.99, p: int = 1):
    return torch.mean(F.relu(input.abs() - tau) ** p)


def _stft_pair(input: torch.Tensor, target: torch.Tensor, nfft: int, hop_fr: float):
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
    return Y, S


def _prepare_mr_terms(
    input: torch.Tensor,
    target: torch.Tensor,
    nfft: int,
    hop_fr: float,
    gamma: float,
    pcs: bool,
):
    Y, S = _stft_pair(input, target, nfft, hop_fr)

    if not pcs:
        Y = use_pcs(Y, nfft)

    Y_abs = Y.abs()
    S_abs = S.abs()

    if (gamma != 1) and (not pcs):
        Y_abs = Y_abs.clamp_min(1e-12).pow(gamma)
        S_abs = S_abs.clamp_min(1e-12).pow(gamma)

    return Y, S, Y_abs, S_abs


def hybrid_loss(pred_stft, true_stft, n_fft):
    pred_stft_real, pred_stft_imag = pred_stft[:,:,:,0], pred_stft[:,:,:,1]
    true_stft_real, true_stft_imag = true_stft[:,:,:,0], true_stft[:,:,:,1]
    pred_mag = torch.sqrt(pred_stft_real**2 + pred_stft_imag**2 + 1e-12)
    true_mag = torch.sqrt(true_stft_real**2 + true_stft_imag**2 + 1e-12)
    pred_real_c = pred_stft_real / (pred_mag**(0.7))
    pred_imag_c = pred_stft_imag / (pred_mag**(0.7))
    true_real_c = true_stft_real / (true_mag**(0.7))
    true_imag_c = true_stft_imag / (true_mag**(0.7))
    real_loss = nn.MSELoss()(pred_real_c, true_real_c)
    imag_loss = nn.MSELoss()(pred_imag_c, true_imag_c)
    mag_loss = nn.MSELoss()(pred_mag**(0.3), true_mag**(0.3))
    
    y_pred = torch.istft(pred_stft_real+1j*pred_stft_imag, n_fft, n_fft // 2, n_fft, window=vorbis_window(n_fft).to(pred_stft.device))
    y_true = torch.istft(true_stft_real+1j*true_stft_imag, n_fft, n_fft // 2, n_fft, window=vorbis_window(n_fft).to(pred_stft.device))
    y_true = torch.sum(y_true * y_pred, dim=-1, keepdim=True) * y_true / (torch.sum(torch.square(y_true),dim=-1,keepdim=True) + 1e-8)

    sisnr =  - torch.log10(torch.norm(y_true, dim=-1, keepdim=True)**2 / (torch.norm(y_pred - y_true, dim=-1, keepdim=True)**2+1e-8) + 1e-8).mean()

    return 30*(real_loss + imag_loss) + 70*mag_loss + sisnr

def loss_compressed_MR(input: torch.Tensor, target: torch.Tensor, gamma: float = 1.0, nffts: list = None, hop_fr: float = 0.25, low_freq_ratio: float = 0.25, pcs: bool = False, lambd=0.3) -> torch.Tensor:
    if nffts is None:
        nffts = [1024, 512, 256]
    loss = torch.zeros((), device=input.device, dtype=input.dtype)
    for nfft in nffts:
        Y, S, Y_abs, S_abs = _prepare_mr_terms(input, target, nfft, hop_fr, gamma, pcs)

        loss += (1 - lambd) * torch.mean((Y_abs - S_abs).abs() ** 2) + lambd * torch.mean(torch.abs(Y_abs * (Y / (torch.abs(Y) + 1e-9)) - S_abs * (S / (torch.abs(S) + 1e-9))) ** 2)

    return (loss / len(nffts)).mean()


def loss_MR_half(input: torch.Tensor, target: torch.Tensor, gamma: float = 1.0, nffts: list = None, hop_fr: float = 0.25, low_freq_ratio: float = 0.25, pcs: bool = False) -> torch.Tensor:
    if nffts is None:
        nffts = [1024, 512, 256]
    loss = torch.zeros((), device=input.device, dtype=input.dtype)
    for nfft in nffts:
        Y, S, Y_abs, S_abs = _prepare_mr_terms(input, target, nfft, hop_fr, gamma, pcs)
        ind_half = Y.shape[1] // 2

        loss += _compute_mr(Y[:, :ind_half, :], Y_abs[:, :ind_half, :], S[:, :ind_half, :], S_abs[:, :ind_half, :]) + _compute_mr(Y[:, ind_half:, :], Y_abs[:, ind_half:, :], S[:, ind_half:, :], S_abs[:, ind_half:, :])
    return loss / len(nffts) 


def loss_MR(input: torch.Tensor, target: torch.Tensor, gamma: float = 1.0, nffts: list = None, hop_fr: float = 0.25, low_freq_ratio: float = 0.25, pcs: bool = False) -> torch.Tensor:
    if nffts is None:
        nffts = [1024, 512, 256]
    loss = torch.zeros((), device=input.device, dtype=input.dtype)
    for nfft in nffts:
        Y, S, Y_abs, S_abs = _prepare_mr_terms(input, target, nfft, hop_fr, gamma, pcs)
        low_sub_band_0 = int((nfft // 2 + 1) * 0.1)
        low_sub_band_1 = int((nfft // 2 + 1) * 0.25)
        low_sub_band_2 = int((nfft // 2 + 1) * 0.5)
        loss += _compute_mr(Y, Y_abs, S, S_abs)

    return loss / len(nffts) 

def loss_MR_low_bins(input: torch.Tensor, target: torch.Tensor, gamma: float = 1.0, nffts: list = None, hop_fr: float = 0.25, low_freq_ratio: float = 0.25, pcs: bool = False) -> torch.Tensor:
    if nffts is None:
        nffts = [1024, 512, 256]
    loss = torch.zeros((), device=input.device, dtype=input.dtype)
    for nfft in nffts:
        Y, S = _stft_pair(input, target, nfft, hop_fr)

        Y = Y[:, Y.shape[1] // 24, :]
        S = S[:, S.shape[1] // 24, :]

        if not pcs:
            Y = use_pcs(Y, nfft)
            
        Y_abs = Y.abs()
        S_abs = S.abs()

        if (gamma != 1) and (not pcs):
            Y_abs = Y_abs.clamp_min(1e-12).pow(gamma)
            S_abs = S_abs.clamp_min(1e-12).pow(gamma)

        low_sub_band_0 = int((nfft // 2 + 1) * 0.1)
        low_sub_band_1 = int((nfft // 2 + 1) * 0.25)
        low_sub_band_2 = int((nfft // 2 + 1) * 0.5)
        loss += _compute_mr(Y, Y_abs, S, S_abs)

    return loss / len(nffts)


def loss_MR_PCS(input: torch.Tensor, target: torch.Tensor, nffts: list = None, hop_fr: float = 0.25) -> torch.Tensor:
    if nffts is None:
        nffts = [1024, 512, 256]
    loss = torch.zeros((), device=input.device, dtype=input.dtype)
    for nfft in nffts:
        Y, S = _stft_pair(input, target, nfft, hop_fr)

        Y = use_pcs(Y, n_fft=nfft)

        Y_abs = Y.abs()
        S_abs = S.abs()

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
    loss = torch.zeros((), device=input.device, dtype=input.dtype)
    for seg in lens:
        input_chunks = torch.split(input, seg, dim=1)
        target_chunks = torch.split(target, seg, dim=1)
        assert len(input_chunks) == len(target_chunks), f"{len(input_chunks)} != {len(target_chunks)}"
        loss_interm = torch.zeros((input.shape[0],), device=input.device, dtype=input.dtype)
        for in_ch, tg_ch in zip(input_chunks, target_chunks):
            cossim = -F.cosine_similarity(in_ch, tg_ch) + 1
            loss_interm += cossim
        loss += torch.sum(loss_interm, dim=0) / loss_interm.shape[0] / len(input_chunks)
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

    return loss_mr_w + 2 * loss_mr


if __name__ == '__main__':

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
