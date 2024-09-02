# jax-Diffusion-svc-v2

## first step
put your audios in ./dataset_raw
run prepare_new.sh
modify configs/base.yaml data_loader dataset_path
## second step
run train.py
## infer
run sampling.py