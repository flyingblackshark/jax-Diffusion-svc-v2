import os
import librosa
import argparse
import numpy as np
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
from functools import partial
import jax
MAX_LENGTH = 16000 * 30
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
import multiprocessing
from multiprocessing import get_context
WIN_SIZE = 1024
HOP_SIZE = 160
N_FFT = 1024
NUM_MELS = 128
f0_min = 80.
f0_max = 880.
mel_basis = librosa_mel_fn(sr=16000, n_fft=N_FFT, n_mels=NUM_MELS, fmin=0, fmax=8000)
mel_basis = np.asarray(mel_basis,dtype=np.float32)
def get_f0(wav,model,params):
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
def load_audio(name,wavPath,spks,model,params):
        name = name[:-4]
        wav,_ =  librosa.load(f"{wavPath}/{spks}/{name}.wav", sr=16000, mono=True)
        test_shape = jax.eval_shape(partial(get_f0,model=model,params=params),jax.ShapeDtypeStruct((1,wav.shape[0]), jnp.float32))
        #batch_length_arr.append(test_shape.shape[1])
        #file_name_arr.append(name)
        measure_length = test_shape.shape[1]
        return (wav , measure_length , name)
def write_result(name,data,length,outPath,spks):
    np.save(f"{outPath}/{spks}/{name}.pit",data[:length])

def batch_process_f0(files,batch_size,outPath,wavPath,spks,mesh):
    x_sharding = NamedSharding(mesh,PartitionSpec('data'))
    model,params = load_model()
    f0_result = []
    pool = get_context("spawn").Pool(processes=8)
    #(audio_data ,estimated_length,name) 
    res = pool.map(partial(load_audio,spks=spks,wavPath=wavPath,model=model,params=params), files)
    jitted_get_f0 = jax.jit(partial(get_f0,model=model,params=params), in_shardings=(x_sharding),out_shardings=x_sharding)

    audio_data = list(map(lambda a:a[0],res))
    length_list = list(map(lambda a:a[1],res))
    name_list = list(map(lambda a:a[2],res))
    for i in range(len(res)):
        audio_data[i] = jnp.pad(audio_data[i],(0,MAX_LENGTH-audio_data[i].shape[0]))
    i = 0
    while i * batch_size < len(audio_data) :
        print(f"{(i+1)*batch_size}/{len(files)}")
        batch_data = audio_data[i*batch_size:(i+1)*batch_size]
        B_padding = batch_size - len(batch_data)
        batch_data = jnp.asarray(batch_data)
        batch_data = jnp.pad(batch_data,((0,B_padding),(0,0)))
        batch_f0 = jitted_get_f0(batch_data)
        batch_f0 = batch_f0[:batch_size-B_padding]
        f0_result.extend(batch_f0)
        i+=1
    f0_result = list(map(np.asarray,f0_result))
    pool.starmap(partial(write_result,outPath=outPath,spks=spks), zip(name_list,f0_result,length_list))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wav", help="wav", dest="wav", required=True)
    parser.add_argument("-o", "--out", help="out", dest="out", required=True)
    parser.add_argument("-bs", "--batch_size",type=int, default=4)

    args = parser.parse_args()
    device_mesh = mesh_utils.create_device_mesh((jax.local_device_count(),))
    mesh = Mesh(devices=device_mesh, axis_names=('data'))
    os.makedirs(args.out, exist_ok=True)
    wavPath = args.wav
    outPath = args.out
    batch_size = args.batch_size
    spk_files = {}
    for spks in os.listdir(wavPath):
        if os.path.isdir(f"{wavPath}/{spks}"):
            os.makedirs(f"{outPath}/{spks}", exist_ok=True)
            files = [f for f in os.listdir(f"{wavPath}/{spks}") if f.endswith(".wav")]
            batch_process_f0(files,batch_size,outPath,wavPath,spks,mesh)
    