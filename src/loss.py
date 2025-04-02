import torch
import torch.nn.functional as F


def loss_MR(input: torch.Tensor, target: torch.Tensor, gamma: float = 0.3, nffts: list = None,
            hop_fr: float = 0.25):
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
        Y_abs = Y.abs()
        S_abs = S.abs()
        if gamma != 1:
            Y_abs = Y_abs.clamp_min(1e-12).pow(gamma)
            S_abs = S_abs.clamp_min(1e-12).pow(gamma)
        loss += F.mse_loss(Y_abs, S_abs)
    #     print(loss.shape, loss, f' !!!!!!!!!lossMR for {nfft}!!!!!!!!!!!!')
    # print(loss.shape, loss, ' !!!!!!!!!lossMR!!!!!!!!!!!!')
    return loss


def loss_MR_w(input: torch.Tensor, target: torch.Tensor, lens: list = None):
    if lens is None:
        lens = [4064, 2032, 1016, 508]
    # print(input.isnan().any(), target.isnan().any(), ' inputs of MRw loss')
    loss = torch.zeros((), device=input.device, dtype=input.dtype)
    for seg in lens:
        input_chunks = torch.split(input, seg, dim=1)
        target_chunks = torch.split(target, seg, dim=1)
        assert len(input_chunks) == len(target_chunks)
        loss_interm = torch.zeros((input.shape[0],), device=input.device, dtype=input.dtype)
        for in_ch, tg_ch in zip(input_chunks, target_chunks):
            # print(in_ch.shape, tg_ch.shape, ' !!!!!!!!!losses chunks!!!!!!!')
            # inmax = in_ch.max(dim=1, keepdim=True)[0]
            # in_ch = in_ch/inmax
            # tgmax = tg_ch.max(dim=1, keepdim=True)[0]
            # tg_ch = tg_ch/tgmax
            cossim = -F.cosine_similarity(in_ch, tg_ch)
            loss_interm += cossim
        # print(loss_interm.shape, ' cossim loss')
        loss += torch.sum(loss_interm, dim=0)/loss_interm.shape[0] / len(input_chunks)
        # print(loss, f' !!!!!!!!!lossMRw for {seg}!!!!!!!!!!!!')
    # print(loss.shape, loss,  ' !!!!!!!!!lossMRw!!!!!!!!!!!!')
    return loss


def loss_tot(input: torch.Tensor, target: torch.Tensor, gamma: float = 0.3, nffts: list = None,
             hop_fr: float = 0.75, lens: list = None):
    return loss_MR_w(input, target, lens) + 2 * loss_MR(input, target, gamma, nffts, hop_fr)

