
import os
import librosa
import argparse
import numpy as np
import jax.numpy as jnp
import jax
from utils import Volume_Extractor
import multiprocessing
from multiprocessing import get_context
from functools import partial
import time
volume_extractor = Volume_Extractor(
    hop_size=512,
    block_size=512,
    model_sampling_rate=44100
)
def extract_volume(audio, sr=44100, threhold=-60.0):
    volume = volume_extractor.extract(audio,sr=sr)
    return volume

def load_audio_and_process(name,wavPath,spks,outPath):
        name = name[:-4]
        wav,_ =  librosa.load(f"{wavPath}/{spks}/{name}.wav", sr=44100, mono=True)
        volume = extract_volume(wav)
        np.save(f"{outPath}/{spks}/{name}.vol",volume)


def process_vol(files,outPath,wavPath,spks):
    start_time = time.time()
    pool = get_context("spawn").Pool(processes=8)
    pool.map(partial(load_audio_and_process,spks=spks,wavPath=wavPath,outPath=outPath), files)
    print("Elapsed time: {:.2f} sec".format(time.time() - start_time))


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
    