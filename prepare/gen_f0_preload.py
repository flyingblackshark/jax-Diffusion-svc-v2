import os
import librosa
import argparse
import numpy as np
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
from functools import partial
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
import multiprocessing
from multiprocessing import get_context
from utils import get_f0
import time
MAX_LENGTH = 16000 * 30
def load_audio(name,wavPath,spks):
        name = name[:-4]
        wav,_ =  librosa.load(f"{wavPath}/{spks}/{name}.wav", sr=16000, mono=True)
        #test_shape = jax.eval_shape(partial(get_f0,model=model,params=params),jax.ShapeDtypeStruct((1,wav.shape[0]), jnp.float32))
        measure_length = wav.shape[0]//160 #test_shape.shape[1]
        return (wav , measure_length , name)
def write_result(name,data,length,outPath,spks):
    np.save(f"{outPath}/{spks}/{name}.pit",data[:length])

def batch_process_f0(files,batch_size,outPath,wavPath,spks,mesh):
    
    x_sharding = NamedSharding(mesh,PartitionSpec('data'))
    model,params = load_model()

    pool = get_context("spawn").Pool(processes=8)
    #(audio_data ,estimated_length,name) 
    
    jitted_get_f0 = jax.jit(partial(get_f0,model=model,params=params), in_shardings=(x_sharding),out_shardings=x_sharding)
    global_batch_size = 1024
    total_rounds = len(files) // global_batch_size + 1
    for i in range(total_rounds):
        batch_files = files[i*global_batch_size:(i+1)*global_batch_size]
        f0_result = []
        res = pool.map(partial(load_audio,spks=spks,wavPath=wavPath), batch_files)
        audio_data = list(map(lambda a:a[0],res))
        length_list = list(map(lambda a:a[1],res))
        name_list = list(map(lambda a:a[2],res))
        for j in range(len(res)):
            audio_data[j] = np.pad(audio_data[j],(0,MAX_LENGTH-audio_data[j].shape[0]))
        j = 0
        while j * batch_size < len(audio_data) :
            print(f"{(j+1)*batch_size}/{len(batch_files)}")
            batch_data = audio_data[j*batch_size:(j+1)*batch_size]
            B_padding = batch_size - len(batch_data)
            batch_data = jnp.asarray(batch_data)
            batch_data = jnp.pad(batch_data,((0,B_padding),(0,0)))
            batch_f0 = jitted_get_f0(batch_data)
            batch_f0 = batch_f0[:batch_size-B_padding]
            f0_result.extend(batch_f0)
            j+=1
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
            start_time = time.time()
            files = [f for f in os.listdir(f"{wavPath}/{spks}") if f.endswith(".wav")]
            batch_process_f0(files,batch_size,outPath,wavPath,spks,mesh)
            print("Elapsed time: {:.2f} sec".format(time.time() - start_time))
    