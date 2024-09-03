import os
import sys

from naive import Unit2MelNaive
from sampling import rectified_flow_sample
from train import Trainer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax.numpy as jnp
import jax
import numpy as np

import argparse
import orbax.checkpoint as ocp
from flax.training.train_state import TrainState
from omegaconf import OmegaConf
from scipy.io.wavfile import write

import librosa
from transformers import FlaxAutoModel
import optax
import audax.core
import audax.core.functional
import audax.core.stft
import jax.numpy as jnp
import audax
from librosa.filters import mel as librosa_mel_fn
from jax_fcpe.utils import load_model
import jax
from jax.experimental.compilation_cache import compilation_cache as cc
cc.set_cache_dir("./jax_cache")
jax.config.update('jax_platform_name', 'cpu')

def dynamic_range_compression_jax(x, C=1, clip_val=1e-5):
    return jnp.log(jnp.clip(x,min=clip_val) * C)
def get_mel(y, keyshift=0, speed=1, center=False):
    sampling_rate = 44100
    n_mels     = 128 #self.n_mels
    n_fft      = 2048 #self.n_fft
    win_size   = 2048 #self.win_size
    hop_length = 512 #self.hop_length
    fmin       = 40 #self.fmin
    fmax       = 16000 #self.fmax
    clip_val   = 1e-5 #self.clip_val
    mel_basis = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel_basis = jnp.asarray(mel_basis,dtype=jnp.float32)
    factor = 2 ** (keyshift / 12)       
    n_fft_new = int(np.round(n_fft * factor))
    win_size_new = int(np.round(win_size * factor))
    hop_length_new = int(np.round(hop_length * speed))
    hann_window= jnp.hanning(win_size_new)
    pad_left = (win_size_new - hop_length_new) //2
    pad_right = max((win_size_new - hop_length_new + 1) //2, win_size_new - y.shape[-1] - pad_left)
    y = jnp.pad(y, ((0,0),(pad_left, pad_right)))
    spec = audax.core.stft.stft(y,n_fft_new,hop_length_new,win_size_new,hann_window,onesided=True,center=False)
    spec = jnp.sqrt(spec.real**2 + spec.imag**2 + (1e-9))
    if keyshift != 0:
        size = n_fft // 2 + 1
        resize = spec.size(1)
        if resize < size:
            spec = jnp.pad(spec, ((0, 0),(0, size-resize)))
        spec = spec[:, :size, :] * win_size / win_size_new   
    spec = spec.transpose(0,2,1)
    spec = jnp.matmul(mel_basis, spec)
    spec = dynamic_range_compression_jax(spec, clip_val=clip_val)
    return spec.transpose(0,2,1)
def get_f0(wav):
    WIN_SIZE = 1024
    HOP_SIZE = 160
    N_FFT = 1024
    NUM_MELS = 128
    f0_min = 80.
    f0_max = 880.
    mel_basis = librosa_mel_fn(sr=16000, n_fft=N_FFT, n_mels=NUM_MELS, fmin=0, fmax=8000)
    mel_basis = jnp.asarray(mel_basis,dtype=jnp.float32)
    model,params = load_model()
    wav = jnp.asarray(wav)
    window = jnp.hanning(WIN_SIZE)
    pad_size = (WIN_SIZE-HOP_SIZE)//2
    wav = jnp.pad(wav, ((0,0),(pad_size, pad_size)),mode="reflect")
    spec = audax.core.stft.stft(wav,N_FFT,HOP_SIZE,WIN_SIZE,window,onesided=True,center=False)
    spec = jnp.sqrt(spec.real**2 + spec.imag**2 + (1e-9))
    spec = spec.transpose(0,2,1)
    mel = jnp.matmul(mel_basis, spec)
    mel = jnp.log(jnp.clip(mel, min=1e-5) * 1)
    mel = mel.transpose(0,2,1)

    def model_predict(mel):
        f0 = model.apply(params,mel,threshold=0.006,method=model.infer)
        uv = (f0 < f0_min).astype(jnp.float32)
        f0 = f0 * (1 - uv)
        return f0
    return model_predict(mel).squeeze(-1)

# def speaker2id(hp,key):
#     import csv
#     reader = csv.reader(open(hp.data.speaker_files, 'r'))
#     for row in reader:
#         if row[0].lower() == key.lower():
#             return int(row[1])
#     raise Exception("Speaker Not Found")

import matplotlib.pylab as plt
def plot_spectrogram_to_numpy(spectrogram):
    spectrogram = np.asarray(spectrogram)
    spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max,top_db=80.)
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    fig.show()
    plt.close()
# def get_spec(wav):
#     window = jnp.hanning(WIN_SIZE)
#     pad_size = (WIN_SIZE-HOP_SIZE)//2
#     wav = jnp.pad(wav, ((pad_size, pad_size)),mode="reflect")
#     spec = audax.core.stft.stft(wav,N_FFT,HOP_SIZE,WIN_SIZE,window,onesided=True,center=False)
#     spec = jnp.sqrt(spec.real**2 + spec.imag**2 + (1e-9))
#     return spec

def main(args):

    hp = OmegaConf.load(args.config)
    #spk = speaker2id(hp,args.spk)
    
    hp = OmegaConf.load("configs/base.yaml")
    rng = jax.random.PRNGKey(hp.train.seed)
    trainer = Trainer(rng, hp, False, False)
    trainer.restore_checkpoint()
    
    wav, sr = librosa.load(args.wave, sr=16000)
    pit = get_f0(np.expand_dims(wav,0)).squeeze(0)
    hubert_model = FlaxAutoModel.from_pretrained("./hubert",from_pt=True, trust_remote_code=True)
    ppg = hubert_model(np.expand_dims(wav,0)).last_hidden_state.squeeze(0)
    ppg = jnp.repeat(ppg,repeats=2,axis=0)
    wav, sr = librosa.load(args.wave, sr=44100)
    mel = get_mel(np.expand_dims(wav,0)).squeeze(0)
    n_frames = int(len(wav) // 512)
    pit = jax.image.resize(pit,shape=(n_frames),method="nearest")
    ppg = jax.image.resize(ppg,shape=(n_frames,ppg.shape[1]),method="nearest")            
    print("pitch shift: ", args.shift)
    if (args.shift == 0):
        pass
    else:
        pit = np.array(pit)
        source = pit[pit > 0]
        source_ave = source.mean()
        source_min = source.min()
        source_max = source.max()
        print(f"source pitch statics: mean={source_ave:0.1f}, \
                min={source_min:0.1f}, max={source_max:0.1f}")
        shift = args.shift
        shift = 2 ** (shift / 12)
        pit = pit * shift
 
    def parallel_infer(pit_i,ppg_i,spk_i):
        init_rngs = {'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(0),'rnorms':jax.random.PRNGKey(0)}
        out_audio = trainer.naive_train_state.apply_fn( {'params': trainer.naive_train_state.params},ppg_i,pit_i,rngs=init_rngs)
        return out_audio
   
    ppg = jnp.asarray(ppg)
    pit = jnp.asarray(pit)
    #spk = jnp.asarray(spk)
    ppg = jnp.expand_dims(ppg,0)
    pit = jnp.expand_dims(pit,0)
    #spk = jnp.expand_dims(spk,0)

    output_mel = parallel_infer(pit,ppg,0).squeeze(0)

    #out_audio = jnp.reshape(frags,[frags.shape[0]*frags.shape[1]*frags.shape[2]])
    #out_audio = np.asarray(out_audio)
    # wav_spec = get_spec(wav).squeeze(0)
    # out_spec = get_spec(out_audio).squeeze(0)
    output_mel = np.asarray(output_mel)
    output_mel = output_mel.transpose(1,0)
    #output_mel = librosa.amplitude_to_db(output_mel, ref=np.max,top_db=80.)
    mel = np.asarray(mel)
    mel = mel.transpose(1,0)
    #mel = librosa.amplitude_to_db(mel, ref=np.max,top_db=80.)
    fig, ax = plt.subplots(nrows=2, ncols=1,figsize=(12, 4))
    
    ax[0].imshow(output_mel, aspect="auto", origin="lower",
                   interpolation='none')
    ax[1].imshow(mel, aspect="auto", origin="lower",
                interpolation='none')
    plt.tight_layout()
    plt.show()
    t_start = 0
    mel = mel.transpose(1,0)
    output_mel = output_mel.transpose(1,0)
    z = t_start * mel + (1 - t_start) * jax.random.normal(jax.random.PRNGKey(0),mel.shape)
    z = rectified_flow_sample(trainer.diff_train_state,z,output_mel,jax.random.PRNGKey(0))

    fig, ax = plt.subplots(nrows=2, ncols=1,figsize=(12, 4))
    
    ax[0].imshow(output_mel, aspect="auto", origin="lower",
                   interpolation='none')
    ax[1].imshow(z, aspect="auto", origin="lower",
                interpolation='none')
    plt.tight_layout()
    plt.show()
    #write("svc_out.wav", 32000, out_audio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,  default="./configs/base.yaml",
                        help="yaml file for config.")
    parser.add_argument('--wave', type=str,  default="./test.wav",
                        help="Path of raw audio.")
    parser.add_argument('--spk', type=str,  default="AURORA",
                        help="Path of speaker.")
    parser.add_argument('--shift', type=int, default=0,
                        help="Pitch shift key.")
    args = parser.parse_args()

    main(args)
