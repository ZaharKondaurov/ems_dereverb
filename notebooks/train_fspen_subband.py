#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyroomacoustics as pra
from scipy.io.wavfile import write

import os
import yaml
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchmetrics.audio import SpeechReverberationModulationEnergyRatio, ShortTimeObjectiveIntelligibility

from einops import rearrange

from src.dataset import SignalDataset, TRUNetDataset
from src.loss import loss_tot, loss_MR, loss_MR_w, phase_losses
from NISQA_s.src.core.model_torch import model_init
from NISQA_s.src.utils.process_utils import process
from torch_stoi import NegSTOILoss
from models.fspen import FullSubPathExtension, SubPathExtension, FullPathExtension

from src.utils import model_eval, model_eval_fspen2x_ver3, mag_phase_stft

import librosa
from IPython.display import Audio

import matplotlib.pyplot as plt

import warnings


# In[2]:


# np.set_printoptions(precision=3)
# torch.set_printoptions(precision=3)
# DATA_DIR = "/opt/software/datasets/urgent26_track2_se_dataset/simulation_train/clean/"  # os.path.join("data", "wav48")
DATA_DIR = os.path.join("data", "wav48") # os.path.join("data", "test_audio")

RIR_DIR = os.path.join("data", "rirs48")
NOISE_DIR = os.path.join("data", "noise48")

CHKP_DIR = "checkpoints"

NISQA_PATH = "NISQA_s/config/nisqa_s.yaml"
np.set_printoptions(precision=3)
torch.set_printoptions(precision=3)


# In[3]:


SEED = 1984

np.random.seed(SEED)
torch.manual_seed(SEED)

gen = torch.Generator()
gen.manual_seed(SEED)


# In[4]:


N_FFTS = 512
HOP_LENGTH = 256 # int(0.01625 * 16_000) # 256
# N_FFTS = 1024
# HOP_LENGTH = 512
SR = 48_000
BATCH_SIZE = 12

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"It's {DEVICE} time!!!")
N_DEVICES = max(torch.cuda.device_count(), 1)


# In[5]:


with open(NISQA_PATH, 'r') as stream:
    nisqa_args = yaml.safe_load(stream)
nisqa_args["ms_n_fft"] = N_FFTS
nisqa_args["hop_length"] = HOP_LENGTH
nisqa_args["ms_win_length"] = N_FFTS
nisqa_args["ckp"] = nisqa_args["ckp"][3:]
nisqa_args["inf_device"] = DEVICE


# In[6]:


nisqa, h0_nisqa, c0_nisqa = model_init(nisqa_args)


# In[7]:


stoi = NegSTOILoss(SR, use_vad=False, do_resample=False).to(DEVICE)


# In[8]:


snr_dict = {1: [-5, ], 5: [-5, ], 7: [-5, ]}
rir_dict = {1: os.path.join("data", "test_rir"), 3: os.path.join("data", "test_rir")}
dataset = TRUNetDataset(DATA_DIR, sr=SR, noise_dir=NOISE_DIR, rir_dir=RIR_DIR, snr=0, rir_proba=1.0, noise_proba=1.0, rir_target=False, return_noise=False, return_rir=False, max_seq_len=SR * 4)

train_dataset, test_dataset = dataset, dataset#  torch.utils.data.random_split(dataset, [0.7, 0.3])


# In[9]:


train_dataset, TRUNetDataset(DATA_DIR, sr=SR, noise_dir=os.path.join("data", "test_noise"), rir_dir=os.path.join("data", "test_rir"), snr=5, rir_proba=0.0, noise_proba=1.0, rir_target=False, return_noise=False, return_rir=False, max_seq_len=SR * 5)
test_dataset = TRUNetDataset(DATA_DIR, sr=SR, noise_dir=os.path.join("data", "test_noise"), rir_dir=os.path.join("data", "test_rir"), snr=5, rir_proba=0.0, noise_proba=1.0, rir_target=False, return_noise=False, return_rir=False, max_seq_len=SR * 5)


# In[10]:


def vorbis_window(winlen, device="cuda"):
    sq = torch.sin(torch.pi/2*(torch.sin(torch.pi/winlen*(torch.arange(winlen)-0.5))**2)).float()
    return sq


# In[11]:


def pad_sequence(batch):
    if not batch:
        return torch.zeros(0), torch.zeros(0)

    input_signal, target_signal, noise, rir = zip(*batch)
        
    max_len_s = max(s.shape[-1] for s in input_signal)
    
    padded_input = torch.zeros(len(input_signal), max_len_s)
    padded_target = torch.zeros(len(target_signal), max_len_s)
    
    for i, s in enumerate(input_signal):
        padded_input[i, :s.shape[-1]] = s
        padded_target[i, :s.shape[-1]] = target_signal[i]

    return padded_input, padded_target

def collate_fn(batch):
    
    padded_input, padded_target = pad_sequence(batch)
    
    # padded_input = padded_input.unfold(-1, 16_000 * 2, 16_000)
    # padded_target = padded_target.unfold(-1, 16_000 * 2, 16_000)
    
    window = vorbis_window(N_FFTS)
    
    padded_input = padded_input.reshape(-1, padded_input.shape[-1])
    input_spec = torch.stft(
            padded_input,
            n_fft=N_FFTS,
            hop_length=HOP_LENGTH,
            # onesided=True,
            win_length=N_FFTS,
            window=window,
            return_complex=True,
            normalized=True,
            center=True
            
        ) 
    to_gt_spec = padded_input.reshape(-1, padded_target.shape[-1])
    gt_spec = torch.stft(
            to_gt_spec,
            n_fft=N_FFTS,
            hop_length=HOP_LENGTH,
            # onesided=True,
            win_length=N_FFTS,
            window=window,
            return_complex=True,
            normalized=True,
            center=True
        ) 
    
    padded_target = padded_target.reshape(-1, padded_target.shape[-1])

    return input_spec, padded_target, gt_spec


# In[12]:


from tqdm import tqdm
import multiprocessing

cores = multiprocessing.cpu_count() # Count the number of cores in a computer
cores


# In[13]:


train_dataloader = DataLoader(train_dataset, batch_size=1 * N_DEVICES, shuffle=False, drop_last=False, collate_fn=collate_fn, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=1 * N_DEVICES, shuffle=False, drop_last=False, collate_fn=collate_fn, num_workers=4)


# In[14]:


from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


# In[15]:


def get_grad_norm(model, dict_grad):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().detach().cpu().item()
            dict_grad[f"grad_norm/{name}"].append(grad_norm)
            dict_grad[f"grad_max/{name}"].append(param.grad.max().detach().cpu().item())
        else:
            dict_grad[f"grad_norm/{name}"].append(None)
            dict_grad[f"grad_max/{name}"].append(None)

    return dict_grad

def train(model, configs, train_loader, optimizer, with_noise=True, with_rir=True, device="cuda", epoch=0, accumulation_step=1, draw_every=1):
    total_train_loss = 0
    # MOS NOI DISC COL LOUD
    total_train_nisqa = torch.zeros(5)
    total_train_srmr = 0
    total_train_stoi = []

    dict_grad_norm = {}
    for name, _ in model.named_parameters():
        dict_grad_norm[f"grad_norm/{name}"] = list()
        dict_grad_norm[f"grad_max/{name}"] = list()
        dict_grad_norm[f"upd_ratio/{name}"] = list()

    model.train()

    for input_spec, gt_signal, gt_spec in tqdm(train_loader, desc="Train model "):
        gt_signal = gt_signal.to(device)
        gt_spec = gt_spec.to(device)

        output, _ = model_eval(model, input_spec, configs, device, hid_size=32)

        window = vorbis_window(N_FFTS).to(device)

        out_wave = torch.istft(output, n_fft=N_FFTS, hop_length=HOP_LENGTH, win_length=N_FFTS,
                               window=window,
                               # onesided=True,
                               return_complex=False,
                               normalized=True,
                               center=True)#, length=gt_signal.shape[-1])
        
        min_l = min(out_wave.shape[-1], gt_signal.shape[-1])
        stoi_score = -stoi(out_wave[..., :min_l], gt_signal[..., :min_l])# .mean()# .detach().cpu()
        loss_mr = loss_MR(out_wave[..., :min_l], gt_signal[..., :min_l], nffts=[128, 256, 512, 1024, 2048, 4096], gamma=0.3)

        loss_ip, loss_gd, loss_iaf = phase_losses(gt_spec.angle(), output.angle())
        loss_pha = loss_ip + loss_gd + loss_iaf

        loss_conv = torch.norm(output - gt_spec) / (torch.norm(gt_spec) + 1e-8)

        loss = loss_mr + loss_conv # + loss_pha * 0.5
        
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if epoch % 5 == 0:
            nisqa_score, _, _ = process(out_wave.detach(), SR, nisqa, h0_nisqa, c0_nisqa, nisqa_args)
            total_train_nisqa += nisqa_score[0].detach().cpu()

        total_train_stoi.append(stoi_score.detach().cpu())
        
        total_train_loss += loss.detach().cpu().item()  

        assert loss.detach().isnan().any().item() is False, "Train loss is NaN"
        
    return (model, optimizer, total_train_loss / len(train_loader), total_train_nisqa / len(train_loader), 
            total_train_srmr / len(train_loader), torch.hstack(total_train_stoi).mean().item(), dict_grad_norm)
            
def evaluate(model, configs, test_loader, with_noise=True, with_rir=True, device="cuda", epoch=0, accumulation_step=1):
    total_test_loss = 0
    total_test_nisqa = torch.zeros(5)
    total_test_srmr = 0
    total_test_stoi = []
    
    model.eval()

    last_out = None
    last_in = None

    with torch.no_grad():
        for input_spec, gt_signal, gt_spec in tqdm(test_loader, desc="Test model "):
            gt_signal = gt_signal.to(device)
            gt_spec = gt_spec.to(device)

            output, _ = model_eval(model, input_spec, configs, device, hid_size=32)

            window = vorbis_window(N_FFTS).to(device)
            out_wave = torch.istft(output, n_fft=N_FFTS, hop_length=HOP_LENGTH, win_length=N_FFTS,
                                   window=window,
                                   return_complex=False,
                                   normalized=True,
                                   center=True)

            min_l = min(out_wave.shape[-1], gt_signal.shape[-1])
            stoi_score = -stoi(out_wave[..., :min_l], gt_signal[..., :min_l])# .mean()# .detach().cpu()    
            loss_mr = loss_MR(out_wave[..., :min_l], gt_signal[..., :min_l], nffts=[128, 256, 512, 1024, 2048, 4096], gamma=0.3)    

            loss_ip, loss_gd, loss_iaf = phase_losses(gt_spec.angle(), output.angle())
            loss_pha = loss_ip + loss_gd + loss_iaf

            loss_conv = torch.norm(output - gt_spec) / (torch.norm(gt_spec) + 1e-8)

            loss = loss_mr + loss_conv # + loss_pha * 0.5

            if epoch % 5 == 0:
                nisqa_score, _, _ = process(out_wave.detach(), SR, nisqa, h0_nisqa, c0_nisqa, nisqa_args)
                total_test_nisqa += nisqa_score[0].detach().cpu()

            total_test_stoi.append(stoi_score.detach().cpu())

            total_test_loss += loss.detach().cpu().item()

            last_out = out_wave
            input_spec = input_spec.to(device)
            last_in = torch.istft(input_spec, n_fft=N_FFTS, hop_length=HOP_LENGTH, win_length=N_FFTS,
                                   window=window,
                                   return_complex=False,
                                   normalized=True,
                                   center=True)

            assert loss.detach().isnan().any().item() is False, "Val loss is NaN"
     
    # if epoch % 1 == 0:
    write('input_sig_part.wav', SR, last_in.cpu().detach().numpy()[0])
    write('output_part.wav', SR, last_out.cpu().detach().numpy()[0])

    return (total_test_loss / len(test_loader), total_test_nisqa / len(test_loader),
            total_test_srmr / len(test_loader), torch.hstack(total_test_stoi).mean().item())
    


# In[16]:


from IPython.display import clear_output

def get_model_name(chkp_folder, model_name=None):
    # Выбираем имя чекпоинта для сохранения
    if model_name is None:
        if os.path.exists(chkp_folder):
            num_starts = len(os.listdir(chkp_folder)) + 1
        else:
            num_starts = 1
        model_name = f'model#{num_starts}'
    else:
        if "#" not in model_name:
            model_name += "#0"
    changed = False
    while os.path.exists(os.path.join(chkp_folder, model_name + '.pt')):
        model_name, ind = model_name.split("#")
        model_name += f"#{int(ind) + 1}"
        changed=True
    if changed:
        warnings.warn(f"Selected model_name was used already! To avoid possible overwrite - model_name changed to {model_name}")
    return model_name

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def learning_loop(
    model,
    configs,
    optimizer,
    train_loader,
    val_loader,
    scheduler=None,
    min_lr=None,
    epochs=10,
    accumulation_step=1,
    val_every=1,
    draw_every=1,
    with_noise=True,
    with_rir=True,
    model_name=None,
    chkp_folder="checkpoints/fspen_chkp",
    plots=None,
    starting_epoch=0,
    device="cuda",
):
    model_name = get_model_name(chkp_folder, model_name)
    
    if plots is None:
        plots = {
            'train loss': [],
            'train NISQA': [],
            'train SRMR': [],
            'train STOI': [],
            'val loss': [],
            'val NISQA': [],
            'val SRMR': [],
            'val STOI': [],
            "learning rate": [],
            "weights_stat": [],
        }

    max_mos = 0

    for epoch in np.arange(1, epochs+1) + starting_epoch:
        print(f'#{epoch}/{epochs}:')
        train_dataset.set_epoch(epoch)
        # print(train_dataset.dataset.snr)
        # print(len(train_dataset.dataset.rir_files))
        test_dataset.set_epoch(epoch)
        plots['learning rate'].append(get_lr(optimizer))
        
        (model, optimizer, train_loss, train_nisqa, train_srmr, train_stoi, weights_stat) = train(model, configs, train_loader, optimizer, with_noise=with_noise, with_rir=with_rir, device=device, epoch=epoch - 1, accumulation_step=accumulation_step, draw_every=draw_every)
        # print(train_nisqa)
        plots['train loss'].append(train_loss)
        if (epoch - 1) % 5 == 0:
            plots['train NISQA'].append(train_nisqa[None, :].cpu())
        else:
            plots['train NISQA'].append(plots['train NISQA'][-1])
        plots['train SRMR'].append(train_srmr)
        plots['train STOI'].append(train_stoi)
        plots["weights_stat"].append(weights_stat)

        # print(weights_stat)
        # for name, value in weights_stat.items():
        #     writer.add_scalar(name, sum(value) / len(value), epoch)

        if not (epoch % val_every):
            # print("validate")
            (val_loss, val_nisqa, val_srmr, val_stoi) = evaluate(model, configs, val_loader, with_noise=with_noise, with_rir=with_rir, epoch=epoch-1, accumulation_step=accumulation_step, device=device)
            plots['val loss'].append(val_loss)
            if (epoch - 1) % 5 == 0:
                plots['val NISQA'].append(val_nisqa[None, :].cpu())
            else:
                plots['val NISQA'].append(plots['val NISQA'][-1])
            plots['val SRMR'].append(val_srmr)
            plots['val STOI'].append(val_stoi)
            
        # Сохраняем модель
        if not os.path.exists(chkp_folder):
            os.makedirs(chkp_folder)
        
        # if max_mos <= val_nisqa[0]:
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'plots': plots,
            },
            os.path.join(chkp_folder, model_name + '.pt'),
        )
        # max_mos = val_nisqa[0]
        
        # Шедулинг
        if scheduler:
            try:
                scheduler.step()
            except:
                scheduler.step(metrics=val_loss)

        if not (epoch % draw_every):
            clear_output(True)

            hh = 4
            ww = 2
            plt_ind = 1
            fig, ax = plt.subplots(hh, ww, figsize=(25, 12))
            fig.suptitle(f'#{epoch}/{epochs}:')


            plt.subplot(hh, ww, plt_ind)
            plt.title('Learning rate')
            plt.plot(plots["learning rate"], 'b.-', label='lr', alpha=0.7)
            plt.legend()
            plt_ind += 1

            plt.subplot(hh, ww, plt_ind)
            plt.title('Loss')
            plt.plot(np.arange(1, epoch + 1), plots['train loss'], 'r.-', label='train', alpha=0.7)
            plt.plot(np.arange(1, epoch + 1), plots['val loss'], 'g.-', label='val', alpha=0.7)
            plt.grid()
            plt.legend()
            plt_ind += 1
            
            plt.subplot(hh, ww, plt_ind)
            plt.title('SRMR')
            plt.plot(np.arange(1, epoch + 1), plots['train SRMR'], 'r.-', label='train', alpha=0.7)
            plt.plot(np.arange(1, epoch + 1), plots['val SRMR'], 'g.-', label='val', alpha=0.7)
            plt.grid()
            plt.legend()
            plt_ind += 1
            
            plt.subplot(hh, ww, plt_ind)
            plt.title('STOI')
            plt.plot(np.arange(1, epoch + 1), plots['train STOI'], 'r.-', label='train', alpha=0.7)
            plt.plot(np.arange(1, epoch + 1), plots['val STOI'], 'g.-', label='val', alpha=0.7)
            plt.grid()
            plt.legend()
            plt_ind += 1

            nisqa_plot = torch.cat(plots['train NISQA'])
            # if len(nisqa_plot.shape) == 1:
            #     nisqa_plot = nisqa_plot[None, :]
            # print(nisqa_plot.shape)
            plt.subplot(hh, ww, plt_ind)
            plt.title('Train NISQA')
            plt.plot(np.arange(1, epoch + 1), nisqa_plot[:, 0], '.-', label='MOS', alpha=0.7, markersize=20, color="blue")
            plt.plot(np.arange(1, epoch + 1), nisqa_plot[:, 1], '.-', label='NOI', alpha=0.7, markersize=20, color="red")
            plt.plot(np.arange(1, epoch + 1), nisqa_plot[:, 2], '.-', label='DISC', alpha=0.7, markersize=20, color="green")
            plt.plot(np.arange(1, epoch + 1), nisqa_plot[:, 3], '.-', label='COL', alpha=0.7, markersize=20, color="yellow")
            plt.plot(np.arange(1, epoch + 1), nisqa_plot[:, 4], '.-', label='LOUD', alpha=0.7, markersize=20, color="pink")
            plt.grid()
            plt.legend()
            plt_ind += 1

            nisqa_plot = torch.cat(plots['val NISQA'], dim=0)
            # if len(nisqa_plot.shape) == 1:
            #     nisqa_plot = nisqa_plot[None, :]
            plt.subplot(hh, ww, plt_ind)
            plt.title('Val NISQA')
            plt.plot(np.arange(1, epoch + 1), nisqa_plot[:, 0], '.-', label='MOS', alpha=0.7, markersize=20, color="blue")
            plt.plot(np.arange(1, epoch + 1), nisqa_plot[:, 1], '.-', label='NOI', alpha=0.7, markersize=20, color="red")
            plt.plot(np.arange(1, epoch + 1), nisqa_plot[:, 2], '.-', label='DISC', alpha=0.7, markersize=20, color="green")
            plt.plot(np.arange(1, epoch + 1), nisqa_plot[:, 3], '.-', label='COL', alpha=0.7, markersize=20, color="yellow")
            plt.plot(np.arange(1, epoch + 1), nisqa_plot[:, 4], '.-', label='LOUD', alpha=0.7, markersize=20, color="pink")
            plt.grid()
            plt.legend()
            plt_ind += 1

            plt.show()
            # display(fig)
                        
        # if min_lr and get_lr(optimizer) <= min_lr:
        #     print(f'Learning process ended with early stop after epoch {epoch}')
        #     break

    writer.close()
    return model, optimizer, plots


# In[17]:


from torch.optim import Adam, AdamW
from src.fspen_configs import TrainConfig48kHzEnc2x_ver2, TrainConfig48kHzEnc2x_enc_ext, TrainConfig48kHzEnc2x_ver2_sub, TrainConfig48kHzEnc2x_ver2_full, TrainConfig
configs = TrainConfig()
# print(sum(configs.bands_num_in_groups), configs.dual_path_extension["num_modules"])
fspen = FullSubPathExtension(configs=configs).to(DEVICE) # TRUNet(nfft=N_FFTS, hop=HOP_LENGTH).cuda()

optimizer = AdamW(fspen.parameters(), lr=5e-2, weight_decay=1e-5)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.1, last_epoch=-1)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, cooldown=1, patience=3, threshold=0.1, mode="min", threshold_mode="abs")


# In[18]:


from src.utils import model_num_params

_, _ = model_num_params(fspen)


# In[19]:


fspen, optimizer, plots = learning_loop(fspen, configs, optimizer, train_dataloader, test_dataloader, scheduler, draw_every=1, epochs=50, accumulation_step=1, min_lr=1e-8, with_noise=False, with_rir=False, model_name="TrainConfig_OG")


# In[ ]:


import pandas as pd

df = pd.DataFrame(plots['weights_stat'][1])
df.head()


# In[ ]:


df.loc[:, [col for col in df.columns if ("dual" in col) and ("upd" in col)]].mean(axis=1).head(10)


# In[ ]:


plots["weights_stat"][0]


# In[ ]:


# state_d = torch.load(os.path.join(CHKP_DIR, "fspen_chkp", "fspen_subband_48khz#2.pt"), weights_only=False)

# fspen.load_state_dict(state_d["model_state_dict"])
# optimizer.load_state_dict(state_d["optimizer_state_dict"])
# scheduler.load_state_dict(state_d["scheduler_state_dict"])
# plots = state_d["plots"]


# In[ ]:


def get_metrics(model, loader, device="cpu"):
    model.eval()
    
    model = model.to(device)
    
    nisqa_scores = []
    srmr_scores = []
    with torch.no_grad():
        for signal in tqdm(loader):
            signal = signal.to(device)
            window = vorbis_window(N_FFTS).to(device)
    
            spec = torch.stft(
                signal,
                n_fft=N_FFTS,
                hop_length=HOP_LENGTH,
                # onesided=True,
                win_length=N_FFTS,
                window=window,
                return_complex=True,
                normalized=True,
                center=True
            ) 
            
            # abs_spectrum = spec.abs()
            # input_spec = torch.permute(torch.view_as_real(spec), dims=(0, 2, 3, 1))
            # batch, frames, channels, frequency = input_spec.shape
            
            # abs_spectrum = torch.permute(abs_spectrum, dims=(0, 2, 1))
            # abs_spectrum = torch.reshape(abs_spectrum, shape=(batch, frames, 1, frequency))
            
            # hidden_state = [[torch.zeros(1, batch * 64, 16 // 8, device=input_spec.device) for _ in range(8)] for _ in range(3)]

            # output, _ = model(input_spec, abs_spectrum, hidden_state)
    
            # output = torch.permute(output, dims=(0, 3, 1, 2))
            # output[..., 0] = torch.expm1(output[..., 0])
            # output = torch.view_as_complex(output)
            
            output, _ = model_eval(model, spec, device)

            window = vorbis_window(N_FFTS).to(device)
            output = torch.istft(output, n_fft=N_FFTS, hop_length=HOP_LENGTH, win_length=N_FFTS,
                                   window=window,
                                   # onesided=True,
                                   return_complex=False,
                                   normalized=True,
                                   center=True)
            
            # resampler = Resample(SR, 48_000)
            # output = resampler(output)
            
            nisqa_score, _, _ = process(output.detach().cpu(), SR, nisqa, h0_nisqa, c0_nisqa, nisqa_args)
            # srmr_score = srmr(output.detach())
            nisqa_scores.append(nisqa_score[0])
            # srmr_scores.append(srmr_score)

    result = {"nisqa": nisqa_scores}
        
    return result


# In[ ]:


def pad_sequence(batch):
    if not batch:
        return torch.zeros(0), torch.zeros(0)

    input_signal, target_signal, noise, rir = zip(*batch)
    # print(input_signal[0].shape)
    max_len_s = max(s.shape[-1] for s in input_signal)
    
    padded_input = torch.zeros(len(input_signal), max_len_s)
    
    for i, s in enumerate(input_signal):
        padded_input[i, :s.shape[-1]] = s

    return padded_input

def collate_fn(batch):
    
    padded_input = pad_sequence(batch)
        
    padded_input = padded_input.reshape(-1, padded_input.shape[-1])

    return padded_input


# In[ ]:


test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False, collate_fn=collate_fn)


# In[ ]:


metrics = get_metrics(fspen, test_dataloader, device="cuda")


# In[ ]:


print("Final nisqa: ", torch.vstack(metrics["nisqa"]).mean(dim=0))


# In[ ]:


# fspen, optimizer, plots = learning_loop(fspen, optimizer, train_dataloader, test_dataloader, scheduler, draw_every=1, epochs=50, min_lr=1e-8, with_noise=False, with_rir=False, model_name="fspen_subband_48khz", plots=plots, starting_epoch=state_d["epoch"])


# In[ ]:




