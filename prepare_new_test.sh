export PYTHONPATH=$PWD
python prepare/resample.py -w ~/bucket/pub_ds/opencpop_train/ -o ~/bucket/temp/waves-16k -s 16000 -t 120
python prepare/resample.py -w ~/bucket/pub_ds/opencpop_train/ -o ~/bucket/temp/waves-44k -s 44100 -t 120
python prepare/gen_f0_preload.py -w ~/bucket/temp/waves-16k/ -o ~/bucket/temp/pitch
python prepare/gen_hubert_preload.py -w ~/bucket/temp/waves-16k/ -o ~/bucket/temp/hubert
python prepare/gen_mel_preload.py -w ~/bucket/temp/waves-44k/ -o ~/bucket/temp/mel
python prepare/gen_vol_preload.py -w ~/bucket/temp/waves-44k/ -o ~/bucket/temp/vol
python make_dataset_new.py -l ~/bucket/pub_ds/opencpop_train/ -d ~/bucket/temp/ -o ~/bucket/pub_ds_processed