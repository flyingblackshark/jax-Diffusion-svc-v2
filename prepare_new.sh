export PYTHONPATH=$PWD
python prepare/resample.py -w ./dataset_raw -o ./dataset/waves-16k -s 16000 -t 12
python prepare/resample.py -w ./dataset_raw -o ./dataset/waves-44k -s 44100 -t 12
python prepare/gen_f0.py -w dataset/waves-16k/ -o dataset/pitch
python prepare/gen_hubert.py -w dataset/waves-16k/ -o dataset/hubert
#python prepare/gen_spec.py -w dataset/waves-44k/ -o dataset/spec
python prepare/gen_mel.py -w dataset/waves-44k/ -o dataset/mel
python make_dataset_new.py -l ./dataset_raw -d ./dataset -o ./processed