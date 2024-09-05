import os
import librosa
import argparse
import numpy as np
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
import jax
MAX_LENGTH = 16000 * 30
from transformers import FlaxWav2Vec2ForCTC
from jax.experimental.compilation_cache import compilation_cache as cc
cc.set_cache_dir("./jax_cache")
jax.config.update('jax_platform_name', 'cpu')
def batch_process_f0(files,batch_size,outPath,wavPath,spks,mesh):
    wav2vec_model = FlaxWav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft",from_pt=True, trust_remote_code=True)
    x_sharding = NamedSharding(mesh,PartitionSpec('data'))
    jitted_wav2vec_model = jax.jit(wav2vec_model, in_shardings=(x_sharding),out_shardings=x_sharding)
    i = 0
   
    batch_data = []
    batch_length = []
    file_name_arr = []
    while i < len(files):
        print(f"{i+1}/{len(files)}")
        file = files[i][:-4]
        file_name_arr.append(file)
        wav, sr = librosa.load(f"{wavPath}/{spks}/{file}.wav", sr=16000, mono=True)
        test_shape = jax.eval_shape(wav2vec_model,jax.ShapeDtypeStruct((1,wav.shape[0]), jnp.float32)).logits
        batch_length.append(test_shape.shape[1])
        wav = np.pad(wav,(0,MAX_LENGTH-wav.shape[0]))
        batch_data.append(wav)
        if len(batch_data) >= batch_size:
            batch_data = np.stack(batch_data)
            batch_hubert = jitted_wav2vec_model(batch_data).logits
            for j in range(batch_hubert.shape[0]):
                file = file_name_arr[j]
                jnp.save(f"./{outPath}/{spks}/{file}.vec",batch_hubert[j,:batch_length[j]])
            batch_data = []
            batch_length = []
            file_name_arr = []
        i+=1
    if len(batch_data) != 0:
        batch_data = np.stack(batch_data)
        b_length = len(batch_data)
        batch_data = np.pad(batch_data,((0,batch_size-b_length),(0,0)))
        batch_hubert = jitted_wav2vec_model(batch_data).last_hidden_state
        batch_hubert = batch_hubert[:b_length]
        for j in range(batch_hubert.shape[0]):
            file = file_name_arr[j]
            jnp.save(f"./{outPath}/{spks}/{file}.vec",batch_hubert[j,:batch_length[j]])

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
            batch_process_f0(files,batch_size,outPath,wavPath,spks,mesh)
    