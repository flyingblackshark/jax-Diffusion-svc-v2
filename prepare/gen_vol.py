
import os
import librosa
import argparse
import numpy as np
import jax.numpy as jnp
import jax
from utils import Volume_Extractor

volume_extractor = Volume_Extractor(
    hop_size=512,
    block_size=512,
    model_sampling_rate=44100
)
def extract_volume(audio, sr=44100, threhold=-60.0):
    volume = volume_extractor.extract(audio,sr=sr)
    return volume

def process_vol(files,outPath,wavPath,spks):
    i = 0
    while i < len(files):
        print(f"{i+1}/{len(files)}")
        file = files[i][:-4]
        wav, sr = librosa.load(f"{wavPath}/{spks}/{file}.wav", sr=44100, mono=True)
        i+=1
        volume = extract_volume(wav)
        volume = jnp.asarray(volume)
        jnp.save(f"{outPath}/{spks}/{file}.vol",volume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wav", help="wav", dest="wav", required=True)
    parser.add_argument("-o", "--out", help="out", dest="out", required=True)
    #parser.add_argument("-bs", "--batch_size",type=int, default=1)

    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    wavPath = args.wav
    outPath = args.out
    #batch_size = args.batch_size
    spk_files = {}
    for spks in os.listdir(wavPath):
        if os.path.isdir(f"{wavPath}/{spks}"):
            os.makedirs(f"{outPath}/{spks}", exist_ok=True)
            files = [f for f in os.listdir(f"{wavPath}/{spks}") if f.endswith(".wav")]
            process_vol(files,outPath,wavPath,spks)
    