export PYTHONPATH=$PWD
python prepare/resample.py -w ~/bucket/dataset_raw -o ~/bucket/dataset/waves-16k -s 16000 -t 12
python prepare/resample.py -w ~/bucket/dataset_raw -o ~/bucket/dataset/waves-44k -s 44100 -t 12
python prepare/gen_f0.py -w ~/bucket/dataset/waves-16k/ -o ~/bucket/dataset/pitch
#python prepare/gen_wav2vec.py -w ~/bucket/dataset/waves-16k/ -o ~/bucket/dataset/vec
python prepare/gen_hubert.py -w ~/bucket/dataset/waves-16k/ -o ~/bucket/dataset/hubert
#python prepare/gen_spec.py -w dataset/waves-44k/ -o dataset/spec
python prepare/gen_mel.py -w ~/bucket/dataset/waves-44k/ -o ~/bucket/dataset/mel
python prepare/gen_vol.py -w ~/bucket/dataset/waves-44k/ -o ~/bucket/dataset/vol
python make_dataset_new.py -l ~/bucket/dataset_raw -d ~/bucket/dataset -o ~/bucket/processed