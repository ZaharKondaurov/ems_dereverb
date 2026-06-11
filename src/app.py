import io
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

import streamlit as st
from streamlit_option_menu import option_menu

import torch
import torchaudio
from torchaudio.transforms import Resample
import tempfile
import librosa

from models.fspen import FullSubPathExtension
from src.fspen_configs import TrainConfig48kHzEnc2x_ver2
from src.dataset import SignalDataset
from src.utils import model_eval

import soundfile as sf

import time
import os

import numpy as np

import matplotlib.pyplot as plt


def melForward(f):
    return 2595 * np.log10(1 + f / 700)


def melInverse(m):
    return (10 ** (m / 2595) - 1) * 700


def vorbis_window(winlen):
    sq = torch.sin(
        torch.pi
        / 2
        * (torch.sin(torch.pi / winlen * (torch.arange(winlen) - 0.5)) ** 2)
    ).float()
    return sq


@st.cache_resource
def load_model(path: str):
    state_d = torch.load(path, weights_only=False, map_location="cpu")

    configs = TrainConfig48kHzEnc2x_ver2()

    model = FullSubPathExtension(configs=configs)

    model.load_state_dict(state_d["model_state_dict"])
    return model


@st.cache_resource
def load_readme():
    with open("README.md", "r", encoding="utf-8") as file:
        file_content = file.read()
    return file_content


def audio_to_bytes(audio, sr):
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="WAV")
    return buffer.getvalue()


selected = option_menu(
    None,
    ["Home", "Experiments"],
    icons=["house", "file-earmark"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#000000"},
        "icon": {"color": "red", "font-size": "25px"},
        "nav-link": {
            "font-size": "25px",
            "text-align": "left",
            "margin": "0px",
            "--hover-color": "#eee",
        },
        "nav-link-selected": {"background-color": "#ff5757"},
    },
)

CHKP_DIR = "checkpoints"
N_FFTS = 1024
HOP_LENGTH = 512
SR = 48_000

if selected == "Home":
    st.title("Speech denoising and dereverberation")

    model_path = os.path.join(
        CHKP_DIR, "fspen_chkp", "TrainConfig48kHzEnc2x_ver2_real#1.pt"
    )

    model = load_model(model_path)
    model.eval()

    st.write("Upload an audio file")

    input_files = st.file_uploader(
        "Choose an audio file", type=["wav", "mp3", "m4a"], accept_multiple_files=True
    )

    with torch.no_grad():
        if input_files:
            input_audio = []
            probs = []

            if len(input_files) > 1:
                st.warning(
                    f"You uploaded {len(input_files)} files. Please upload no more than {1}."
                )
            else:
                if st.button("Start Prediction"):
                    mean_time = 0
                    for ind, uploaded_file in enumerate(input_files):
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".wav"
                        ) as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            tmp_path = tmp_file.name
                        try:
                            signal, signal_sr = torchaudio.load(tmp_path)

                            signal, _ = SignalDataset.normalize_audio(signal)
                            if signal_sr != SR:
                                resampler = Resample(signal_sr, SR)
                                signal = resampler(signal)

                            audio_bytes = audio_to_bytes(signal.reshape(-1), SR)
                            st.write("Input audio")
                            st.audio(audio_bytes, format="audio/wav")

                            window = vorbis_window(N_FFTS)
                            spec = torch.stft(
                                signal,
                                n_fft=N_FFTS,
                                hop_length=HOP_LENGTH,
                                # onesided=True,
                                win_length=N_FFTS,
                                window=window,
                                return_complex=True,
                                normalized=True,
                                center=True,
                            )

                            spec_vis = torch.stft(
                                signal,
                                n_fft=N_FFTS,
                                hop_length=HOP_LENGTH,
                                # onesided=True,
                                win_length=N_FFTS,
                                window=window,
                                return_complex=False,
                                normalized=True,
                                center=True,
                            )

                            spec_vis = spec_vis.norm(dim=-1).pow(2)

                            tGrid = (
                                np.arange(0, spec_vis.shape[2] * HOP_LENGTH, HOP_LENGTH)
                                / SR
                            )
                            fGrid = np.arange(0, N_FFTS / 2 + 0.00001) / (N_FFTS) * SR
                            tt, ff = np.meshgrid(tGrid, fGrid)

                            fig, ax = plt.subplots(figsize=(10, 4))

                            img = ax.pcolormesh(
                                tt,
                                ff,
                                20 * torch.log10(spec_vis.squeeze() + 1e-8),
                                cmap="gist_heat",
                            )

                            ax.set_xlabel("Time, sec", size=20)
                            ax.set_ylabel("Frequency, Hz", size=20)
                            ax.set_yscale(
                                "function", functions=(melForward, melInverse)
                            )
                            fig.colorbar(img, ax=ax)

                            st.pyplot(fig)

                        except Exception as e:
                            st.error(f"Error during preprocessing: {e}")
                            st.stop()

                        with st.spinner("Analyzing audio..."):
                            try:
                                model.eval()
                                with torch.no_grad():
                                    compute_time = time.time()
                                    output, _ = model_eval(model, spec, "cpu")
                                    compute_time = time.time() - compute_time
                            except Exception as e:
                                st.error(f"Prediction error: {e}")
                                st.stop()

                        mean_time += compute_time

                        window = vorbis_window(N_FFTS)

                        out_wave = torch.istft(
                            output,
                            n_fft=N_FFTS,
                            hop_length=HOP_LENGTH,
                            win_length=N_FFTS,
                            window=window,
                            # onesided=True,
                            return_complex=False,
                            normalized=True,
                            center=True,
                        )
                        out_wave = out_wave.reshape(-1)

                        st.write(f"**Compute time:** {compute_time:.4f}s")

                        audio_bytes = audio_to_bytes(out_wave, SR)
                        st.write("Output audio")
                        st.audio(audio_bytes, format="audio/wav")
                        st.download_button(
                            label="Download audio",
                            data=audio_bytes,
                            file_name="result.wav",
                            mime="audio/wav",
                        )

                        spec_vis = torch.view_as_real(output).norm(dim=-1).pow(2)

                        tGrid = (
                            np.arange(0, spec_vis.shape[2] * HOP_LENGTH, HOP_LENGTH)
                            / SR
                        )
                        fGrid = np.arange(0, N_FFTS / 2 + 0.00001) / (N_FFTS) * SR
                        tt, ff = np.meshgrid(tGrid, fGrid)

                        fig, ax = plt.subplots(figsize=(10, 4))

                        img = ax.pcolormesh(
                            tt,
                            ff,
                            20 * torch.log10(spec_vis.squeeze() + 1e-8),
                            cmap="gist_heat",
                        )

                        ax.set_xlabel("Time, sec", size=20)
                        ax.set_ylabel("Frequency, Hz", size=20)
                        ax.set_yscale("function", functions=(melForward, melInverse))
                        fig.colorbar(img, ax=ax)

                        st.pyplot(fig)

else:
    text_content = load_readme()
    st.markdown(text_content, unsafe_allow_html=True)
