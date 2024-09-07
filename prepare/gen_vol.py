
import os
import librosa
import argparse
import numpy as np
import jax.numpy as jnp
import jax
class Volume_Extractor:
    def __init__(self, hop_size=512, block_size=None, model_sampling_rate=None):
        self.block_size = block_size
        self.model_sampling_rate = model_sampling_rate
        self.hop_size = hop_size
        if (self.block_size is not None) or (self.model_sampling_rate is not None):
            assert (self.block_size is not None) and (self.model_sampling_rate is not None)
            self.hop_size_follow_input = True
        else:
            self.hop_size_follow_input = False

    def extract(self, audio, sr=None):  # audio: 1d numpy array
        if sr is not None:
            assert self.hop_size_follow_input
            self.hop_size = self.block_size * sr / self.model_sampling_rate
        n_frames = int(len(audio) // self.hop_size) + 1
        audio2 = audio ** 2
        audio2 = np.pad(audio2, (int(self.hop_size // 2), int((self.hop_size + 1) // 2)), mode='reflect')
        volume = np.array(
            [np.mean(audio2[int(n * self.hop_size): int((n + 1) * self.hop_size)]) for n in range(n_frames)])
        volume = np.sqrt(volume)
        '''
        if isinstance(audio, torch.Tensor):
            n_frames = int(audio.size(-1) // self.hop_size) + 1
            audio2 = audio ** 2
            audio2 = torch.nn.functional.pad(audio2, (int(self.hop_size // 2), int((self.hop_size + 1) // 2)),
                                             mode='reflect')
            audio_frame = torch.nn.functional.unfold(audio2[:, None, None, :], (1, int(self.hop_size)),
                                                     stride=int(self.hop_size))[:, :, :n_frames]
            volume = audio_frame.mean(dim=1)[0]
            volume = torch.sqrt(volume).squeeze().cpu().numpy()
        else:
            n_frames = int(len(audio) // self.hop_size) + 1
            audio2 = audio ** 2
            audio2 = np.pad(audio2, (int(self.hop_size // 2), int((self.hop_size + 1) // 2)), mode='reflect')
            volume = np.array(
                [np.mean(audio2[int(n * self.hop_size): int((n + 1) * self.hop_size)]) for n in range(n_frames)])
            volume = np.sqrt(volume)
        '''
        return volume

    # def get_mask_from_volume(self, volume, threhold=-60.0,device='cpu'):
    #     mask = (volume > 10 ** (float(threhold) / 20)).astype('float')
    #     mask = np.pad(mask, (4, 4), constant_values=(mask[0], mask[-1]))
    #     mask = np.array([np.max(mask[n: n + 9]) for n in range(len(mask) - 8)])
    #     mask = torch.from_numpy(mask).float().to(device).unsqueeze(-1).unsqueeze(0)
    #     mask = upsample(mask, self.block_size).squeeze(-1)
    #     return mask
volume_extractor = Volume_Extractor(
    hop_size=512,
    block_size=512,
    model_sampling_rate=44100
)
def extract_volume(audio, sr=44100, threhold=-60.0):
    volume = volume_extractor.extract(audio, sr)
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
    