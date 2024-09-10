import os
import librosa
import argparse
import numpy as np
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
import jax
from jax.experimental.compilation_cache import compilation_cache as cc
cc.set_cache_dir("jax_cache")
from transformers import FlaxAutoModel
import multiprocessing
from multiprocessing import get_context
from functools import partial
import time

MAX_LENGTH = 16000 * 30

def load_audio(name,wavPath,spks):
        name = name[:-4]
        wav,_ =  librosa.load(f"{wavPath}/{spks}/{name}.wav", sr=16000, mono=True)
        length = wav.shape[0]//320 - 1
        return (wav ,length, name)
def write_result(name,data,length,outPath,spks):
    np.save(f"{outPath}/{spks}/{name}.bert",data[:length])
def get_hubert(x,model):
    return model(x).last_hidden_state
def batch_process_hubert(files,batch_size,outPath,wavPath,spks,mesh):
    
    hubert_model = FlaxAutoModel.from_pretrained("hubert",from_pt=True, trust_remote_code=True)
    x_sharding = NamedSharding(mesh,PartitionSpec('data'))
    jitted_hubert_model = jax.jit(partial(get_hubert,model=hubert_model), in_shardings=(x_sharding),out_shardings=x_sharding)
    global_batch_size = 1024
    total_rounds = len(files) // global_batch_size + 1
    for i in range(total_rounds):
        batch_files = files[i*global_batch_size:(i+1)*global_batch_size]
        hubert_result = []
        pool = get_context("spawn").Pool(processes=8)
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
            batch_hubert = jitted_hubert_model(batch_data)
            batch_hubert = batch_hubert[:batch_size-B_padding]
            hubert_result.extend(batch_hubert)
            j+=1
        hubert_result = list(map(np.asarray,hubert_result))
        pool.starmap(partial(write_result,outPath=outPath,spks=spks), zip(name_list,hubert_result,length_list))
    
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
            batch_process_hubert(files,batch_size,outPath,wavPath,spks,mesh)
            print("Elapsed time: {:.2f} sec".format(time.time() - start_time))
    