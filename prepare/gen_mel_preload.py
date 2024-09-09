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
import jax
from jax.experimental.compilation_cache import compilation_cache as cc
from utils import get_mel
import multiprocessing
from multiprocessing import get_context
from functools import partial
import time
cc.set_cache_dir("jax_cache")
MAX_LENGTH = 30 * 44100

def load_audio(name,wavPath,spks):
        name = name[:-4]
        wav,_ =  librosa.load(f"{wavPath}/{spks}/{name}.wav", sr=44100, mono=True)
        test_shape = jax.eval_shape(get_mel,jax.ShapeDtypeStruct((1,wav.shape[0]), jnp.float32))
        measure_length = test_shape.shape[1]
        return (wav , measure_length , name)
def write_result(name,data,length,outPath,spks):
    np.save(f"{outPath}/{spks}/{name}.mel",data[:length])

def batch_process_mel(files,batch_size,outPath,wavPath,spks,mesh):
    start_time = time.time()
    x_sharding = NamedSharding(mesh,PartitionSpec('data'))
    jitted_get_mel = jax.jit(get_mel, in_shardings=(x_sharding),out_shardings=x_sharding)
    i = 0
    mel_result = []
    pool = get_context("spawn").Pool(processes=8)
    #(audio_data ,estimated_length,name) 
    res = pool.map(partial(load_audio,spks=spks,wavPath=wavPath), files)
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
        batch_mel = jitted_get_mel(batch_data)
        batch_mel = batch_mel[:batch_size-B_padding]
        mel_result.extend(batch_mel)
        i+=1
    mel_result = list(map(np.asarray,mel_result))
    pool.starmap(partial(write_result,outPath=outPath,spks=spks), zip(name_list,mel_result,length_list))
    print("Elapsed time: {:.2f} sec".format(time.time() - start_time))
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
            batch_process_mel(files,batch_size,outPath,wavPath,spks,mesh)
    