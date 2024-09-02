import grain
import numpy as np
import grain.python
import tensorflow as tf


class PadToMaxLength(grain.python.MapTransform):
  """Pads each input to the specified length"""

  def __init__(self, 
               #audio_max_length,
               hubert_max_length,
               f0_max_length,
               spec_max_length):
    #self.audio_max_length = audio_max_length
    self.hubert_max_length = hubert_max_length
    self.f0_max_length = f0_max_length
    self.spec_max_length = spec_max_length

  def map(self, data):
    """map to each element"""

    # def pad_audio(x, max_length):
    #   pad_amount = max(max_length - x.shape[0], 0)
    #   pad_amount = [(0, pad_amount)]
    #   return np.pad(x, pad_amount)
    def pad_hubert(x, max_length):
      pad_amount = max(max_length - x.shape[0], 0)
      return np.pad(x, ((0,pad_amount),(0,0)))
    def pad_f0(x, max_length):
      pad_amount = max(max_length - x.shape[0], 0)
      return np.pad(x, ((0,pad_amount)))
    def pad_mel(x, max_length):
      pad_amount = max(max_length - x.shape[0], 0)
      return np.pad(x, ((0,pad_amount),(0,0)))
    
    #data["audio"] = pad_audio(data["audio"], self.audio_max_length)
    data["hubert_feature"] = pad_hubert(data["hubert_feature"], self.hubert_max_length)
    data["f0_feature"] = pad_f0(data["f0_feature"], self.f0_max_length)
    data["mel_feature"] = pad_mel(data["mel_feature"], self.spec_max_length)
    return data
class SliceToLength(grain.python.RandomMapTransform):
  def __init__(self,segment_size):
    self.segment_size = segment_size
  def random_map(self, data, rng: np.random.Generator):
    idx = rng.integers(0,data["hubert_feature"].shape[0]-self.segment_size-2)
    data["hubert_feature"] = data["hubert_feature"][idx:idx+self.segment_size]
    data["f0_feature"] = data["f0_feature"][idx:idx+self.segment_size]
    data["mel_feature"] = data["mel_feature"][idx:idx+self.segment_size]
    return data

class ParseFeatures(grain.python.MapTransform):
  def __init__(self, hp):
    self.hp = hp
  # def speaker2id(self,key):
  #   import csv
  #   reader = csv.reader(open(self.hp.data.speaker_files, 'r'))
  #   for row in reader:
  #     if row[0].lower() == key:
  #       return int(row[1])
  #   raise Exception("Speaker Not Found")
  def map(self, features):
    def _parse(example):
      parsed = tf.io.parse_example(example, {
        #"audio": tf.io.FixedLenFeature([], dtype=tf.string),
        "f0_feature": tf.io.FixedLenFeature([], dtype=tf.string),
        "hubert_feature": tf.io.FixedLenFeature([], dtype=tf.string),
        "mel_feature": tf.io.FixedLenFeature([], dtype=tf.string),
        "speaker": tf.io.FixedLenFeature([], dtype=tf.string)
        })
      return parsed
    example = _parse(features)
    #audio = tf.io.parse_tensor(example["audio"],tf.float32)
    hubert_feature = tf.io.parse_tensor(example["hubert_feature"],tf.float32)
    f0_feature = tf.io.parse_tensor(example["f0_feature"],tf.float32)
    mel_feature = tf.io.parse_tensor(example["mel_feature"],tf.float32)
    #hubert_feature = tf.repeat(hubert_feature,repeats=2,axis=0) 

    return {
        # "audio": audio,
        # "audio_length":audio.shape[0],
        "hubert_feature": hubert_feature,
        "hubert_length":hubert_feature.shape[0],
        "f0_feature": f0_feature,
        "f0_length": f0_feature.shape[0],
        "mel_feature":mel_feature,
        "mel_length": mel_feature.shape[0],
        #"speaker_id":self.speaker2id(example["speaker"])
    }
  