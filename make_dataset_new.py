import tensorflow as tf
import argparse
import os
import jax
from array_record.python.array_record_module import ArrayRecordWriter
import jax.numpy as jnp
def write_example_to_arrayrecord(example, file_path):
  writer = ArrayRecordWriter(file_path, 'group_size:1')
  writer.write(example.SerializeToString())
  writer.close()  
def process_file(dsPath,spks,listPath,outPath):
    for file in os.listdir(f"{listPath}/{spks}"):
        if file.endswith(".wav"):
            file = file[:-4]
            #wav, sr = librosa.load(f"{dsPath}/waves-32k/{spks}/{file}.wav", sr=32000)
            #spec = jnp.load(f"{dsPath}/spec/{spks}/{file}.spec.npy")
            f0 = jnp.load(f"{dsPath}/pitch/{spks}/{file}.pit.npy")
            vol = jnp.load(f"{dsPath}/vol/{spks}/{file}.vol.npy")
            vol = vol[:-1]
            hubert = jnp.load(f"{dsPath}/hubert/{spks}/{file}.bert.npy")
            #vec = jnp.load(f"{dsPath}/vec/{spks}/{file}.vec.npy")
            mel = jnp.load(f"{dsPath}/mel/{spks}/{file}.mel.npy")
            hubert = jnp.repeat(hubert,repeats=2,axis=0)
            #vec = jnp.repeat(vec,repeats=2,axis=0)
            f0 = jax.image.resize(f0,shape=(mel.shape[0]),method="nearest")
            #vec = jax.image.resize(vec,shape=(mel.shape[0],vec.shape[1]),method="nearest")      
            hubert = jax.image.resize(hubert,shape=(mel.shape[0],hubert.shape[1]),method="nearest")             
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        # 'audio': tf.train.Feature(
                        #     bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(wav).numpy()])),
                        # 'spec_feature': tf.train.Feature(
                        #     bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(spec).numpy()])),
                        'f0_feature': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(f0).numpy()])),
                        # 'wav2vec_feature': tf.train.Feature(
                        #     bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(vec).numpy()])),
                        'hubert_feature': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(hubert).numpy()])),
                        'mel_feature': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(mel).numpy()])),
                        'vol_feature': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(vol).numpy()])),
                        'speaker':tf.train.Feature(bytes_list=tf.train.BytesList(value=[spks.encode('utf-8')]))
                    }
                )
            )
            write_example_to_arrayrecord(example=example,file_path=f"{outPath}/{spks}/{file}.arrayrecord")
            print("write "+f"{outPath}/{spks}/{file}.arrayrecord")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--list", help="list", dest="list", required=True)
    parser.add_argument("-d", "--ds", help="ds", dest="ds", required=True)
    parser.add_argument("-o", "--out", help="out", dest="out", required=True)

    args = parser.parse_args()
    dsPath = args.ds
    outPath = args.out
    listPath = args.list
    os.makedirs(args.out, exist_ok=True)
    
    for spks in os.listdir(listPath):
        if os.path.isdir(f"{listPath}/{spks}"):
            os.makedirs(f"{outPath}/{spks}", exist_ok=True)
            process_file(dsPath,spks,listPath,outPath)