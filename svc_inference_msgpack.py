import os
import sys
import flax
import flax.serialization
from tqdm import tqdm
from models import NaiveV2Diff
from naive import Unit2MelNaive
from sampling import rectified_flow_sample
from train import Trainer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax.numpy as jnp
import jax
import numpy as np

import argparse
from omegaconf import OmegaConf
from scipy.io.wavfile import write

import librosa
from transformers import FlaxAutoModel
import audax.core
import audax.core.functional
import audax.core.stft
import jax.numpy as jnp
import audax
from librosa.filters import mel as librosa_mel_fn
from jax_fcpe.utils import load_model
import jax
from jax.experimental.compilation_cache import compilation_cache as cc
from prepare.utils import get_mel,get_f0,Volume_Extractor
cc.set_cache_dir("./jax_cache")
#jax.config.update('jax_platform_name', 'cpu')
volume_extractor = Volume_Extractor(
    hop_size=512,
    block_size=512,
    model_sampling_rate=44100
)
def extract_volume(audio, sr=44100, threhold=-60.0):
    volume = volume_extractor.extract(audio, sr)
    return volume

# def speaker2id(hp,key):
#     import csv
#     reader = csv.reader(open(hp.data.speaker_files, 'r'))
#     for row in reader:
#         if row[0].lower() == key.lower():
#             return int(row[1])
#     raise Exception("Speaker Not Found")


import matplotlib.pylab as plt
jax.default_matmul_precision='float32'

def main(args):
    
    hp = OmegaConf.load(args.config)
    diff_model = NaiveV2Diff(
        mel_channels=128,
        dim=hp.model_diff.dim,
        use_mlp=True,
        mlp_factor=4,
        num_layers=hp.model_diff.num_layers,
        expansion_factor=2,
        kernel_size=31,
        conv_dropout=hp.model_diff.conv_dropout,
        atten_dropout=hp.model_diff.atten_dropout,
    )
    naive_model = Unit2MelNaive(
                n_spk=hp.model_naive.n_spk,
                out_dims=128,
                use_speaker_encoder=False,
                speaker_encoder_out_channels=256,
                n_layers=hp.model_naive.n_layers,
                n_chans=hp.model_naive.n_chans,
                expansion_factor=hp.model_naive.expansion_factor,
                kernel_size=hp.model_naive.kernel_size,
                num_heads=hp.model_naive.num_heads,
                conv_dropout=hp.model_naive.conv_dropout,
                atten_dropout=hp.model_naive.atten_dropout,
                precision=jax.lax.Precision.DEFAULT)
    #spk = speaker2id(hp,args.spk)
    #rng = jax.random.PRNGKey(hp.train.seed)
    #trainer = Trainer(rng, hp, False, False)

    naive_file = open("naive_params.msgpack", "rb")
    naive_params_bytes = naive_file.read()
    naive_file.close()
    naive_params = flax.serialization.msgpack_restore(naive_params_bytes)

    diff_file = open("diff_params.msgpack", "rb")
    diff_params_bytes = diff_file.read()
    diff_file.close()
    diff_params = flax.serialization.msgpack_restore(diff_params_bytes)

    wav_44k, sr = librosa.load(args.wave, sr=44100,mono=True)
    mel = get_mel(np.expand_dims(wav_44k,0)).squeeze(0)

    wav_16k, sr = librosa.load(args.wave, sr=16000,mono=True)
    vol = extract_volume(wav_44k) 
    f0_model,f0_params = load_model()
    pit = get_f0(np.expand_dims(wav_16k,0),f0_model,f0_params).squeeze(0)
    hubert_model = FlaxAutoModel.from_pretrained("./hubert",from_pt=True, trust_remote_code=True)
    ppg = hubert_model(np.expand_dims(wav_16k,0)).last_hidden_state.squeeze(0)
    ppg = jnp.repeat(ppg,repeats=2,axis=0)

    
    n_frames = int(len(wav_44k) // 512)
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
 
    def parallel_infer(pit_i,ppg_i,spk_i,vol):
        init_rngs = {'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(0),'rnorms':jax.random.PRNGKey(0)}
        out_audio = jax.jit(naive_model.apply)( {'params': naive_params},ppg_i,pit_i,vol,rngs=init_rngs)
        return out_audio
   
    ppg = jnp.asarray(ppg)
    pit = jnp.asarray(pit)
    vol = jnp.asarray(vol)
    #spk = jnp.asarray(spk)
    ppg = jnp.expand_dims(ppg,0)
    pit = jnp.expand_dims(pit,0)
    vol = jnp.expand_dims(vol,0)
    #spk = jnp.expand_dims(spk,0)

    output_mel = parallel_infer(pit,ppg,0,vol).squeeze(0)   
    @jax.jit
    def sample_loop(
        z, t, cond, cfg, dt, rng_key
    ):
        v_cond = diff_model.apply({"params": diff_params}, z, t, cond, rng_key, False)

        z = z - dt * v_cond
        return z
    def rectified_flow_sample(
        z,
        cond,
        rng_key,
        sample_steps: int = 30,
        t_start:float = 0.,
        cfg: float = 2.0,
    ):
        b = z.shape[0]
        dt = 1.0 / sample_steps

        dt = jnp.array([dt] * b).reshape([b] + [1] * (len(z.shape) - 1))


        for i in tqdm(range(sample_steps, 0, -1), desc="Sampling", leave=False):
            t = i * (1 - t_start) / sample_steps
            t = jnp.array([t] * b)

            z = sample_loop(
                z, t, cond, cfg,  dt, rng_key
            )

        return z
    fig, ax = plt.subplots(nrows=3, ncols=1,figsize=(12, 4))

    ax[0].imshow(np.asarray(mel).transpose(1,0), aspect="auto", origin="lower",
                interpolation='none')

    t_start = 0.

    z = t_start * output_mel + (1 - t_start) * jax.random.normal(jax.random.PRNGKey(0),output_mel.shape)

    z = jnp.expand_dims(z,0)
    z = rectified_flow_sample(z,output_mel,jax.random.PRNGKey(0),sample_steps=1000,t_start=t_start)
    z = z.squeeze(0)
    
    ax[1].imshow(np.asarray(output_mel).transpose(1,0), aspect="auto", origin="lower",
                   interpolation='none')
    ax[2].imshow(np.asarray(z).transpose(1,0), aspect="auto", origin="lower",
                interpolation='none')
    plt.tight_layout()
    fig.savefig('test.png')

    config = OmegaConf.load("./vocoder/base.yaml")
    from vocoder.models import Generator
    model = Generator(config)
    params = model.init(jax.random.PRNGKey(0),jnp.ones((1,128,100)),jnp.ones((1,100)))["params"]
    binary_file = open("./vocoder/vocoder.msgpack", "rb")
    bytes_data = binary_file.read()
    binary_file.close()
    params = flax.serialization.from_bytes(params,bytes_data)
    rng = {'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(0),'rnorms':jax.random.PRNGKey(0)}
    z = z.transpose(1,0)
    mel = jnp.expand_dims(z,0)
    wav = jax.jit(model.apply)({"params":params},mel,pit,rngs=rng)
    wav = wav.squeeze(0).squeeze(0)
    write("test_out.wav", 44100, np.asarray(wav))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,  default="./configs/base.yaml",
                        help="yaml file for config.")
    parser.add_argument('--wave', type=str,  default="./test.mp3",
                        help="Path of raw audio.")
    parser.add_argument('--spk', type=str,  default="AURORA",
                        help="Path of speaker.")
    parser.add_argument('--shift', type=int, default=0,
                        help="Pitch shift key.")
    args = parser.parse_args()

    main(args)
