import os
import librosa
import argparse
import numpy as np
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
import jax
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
MAX_LENGTH = 32000 * 30
WIN_SIZE = 1024
HOP_SIZE = 320
N_FFT = 1024
def get_spec(wav):
    window = jnp.hanning(WIN_SIZE)
    pad_size = (WIN_SIZE-HOP_SIZE)//2
    wav = jnp.pad(wav, ((0,0),(pad_size, pad_size)),mode="reflect")
    spec = audax.core.stft.stft(wav,N_FFT,HOP_SIZE,WIN_SIZE,window,onesided=True,center=False)
    spec = jnp.sqrt(spec.real**2 + spec.imag**2 + (1e-6))
    return spec

def batch_process_spec(files,batch_size,outPath,wavPath,spks,mesh):
    x_sharding = NamedSharding(mesh,PartitionSpec('data'))
    jitted_get_spec = jax.jit(get_spec, in_shardings=(x_sharding),out_shardings=x_sharding)
    i = 0
   
    batch_data = []
    batch_length = []
    file_name_arr = []
    while i < len(files):
        print(f"{i+1}/{len(files)}")
        file = files[i][:-4]
        file_name_arr.append(file)
        wav, sr = librosa.load(f"{wavPath}/{spks}/{file}.wav", sr=32000, mono=True)
        test_shape = jax.eval_shape(get_spec,jax.ShapeDtypeStruct((1,wav.shape[0]), jnp.float32))
        batch_length.append(test_shape.shape[1])
        wav = np.pad(wav,(0,MAX_LENGTH-wav.shape[0]))
        batch_data.append(wav)
        i+=1
        if len(batch_data) >= batch_size:
            batch_data = np.stack(batch_data)
            batch_spec = jitted_get_spec(batch_data)
            for j in range(batch_spec.shape[0]):
                file = file_name_arr[j]
                jnp.save(f"./{outPath}/{spks}/{file}.spec",batch_spec[j,:batch_length[j]])
            batch_data = []
            batch_length = []
            file_name_arr = []
    if len(batch_data) != 0:
        batch_data = np.stack(batch_data)
        b_length = len(batch_data)
        batch_data = np.pad(batch_data,((0,batch_size-b_length),(0,0)))
        batch_spec = jitted_get_spec(batch_data)
        batch_spec = batch_spec[:b_length]
        for j in range(batch_spec.shape[0]):
                file = file_name_arr[j]
                jnp.save(f"./{outPath}/{spks}/{file}.spec",batch_spec[j,:batch_length[j]])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wav", help="wav", dest="wav", required=True)
    parser.add_argument("-o", "--out", help="out", dest="out", required=True)
    parser.add_argument("-bs", "--batch_size",type=int, default=1)

    args = parser.parse_args()
    device_mesh = mesh_utils.create_device_mesh((1,))
    mesh = Mesh(devices=device_mesh, axis_names=('data'))
    os.makedirs(args.out, exist_ok=True)
    wavPath = args.wav
    outPath = args.out
    batch_size = args.batch_size
    spk_files = {}
    for spks in os.listdir(wavPath):
        if os.path.isdir(f"./{wavPath}/{spks}"):
            os.makedirs(f"./{outPath}/{spks}", exist_ok=True)
            files = [f for f in os.listdir(f"./{wavPath}/{spks}") if f.endswith(".wav")]
            batch_process_spec(files,batch_size,outPath,wavPath,spks,mesh)
    