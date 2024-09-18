import os
import sys
import flax
import flax.serialization
from train import Trainer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from omegaconf import OmegaConf
import flax
import jax
from jax.experimental.compilation_cache import compilation_cache as cc
cc.set_cache_dir("./jax_cache")


def main(args):
    
    hp = OmegaConf.load(args.config)
    rng = jax.random.PRNGKey(hp.train.seed)
    trainer = Trainer(rng, hp, False, False)
    trainer.restore_checkpoint()
    naive_params = trainer.naive_train_state.params
    diff_params = trainer.diff_train_state.params
    naive_params = flax.serialization.msgpack_serialize(naive_params)
    naive_file = open("naive_params.msgpack", "wb")
    naive_file.write(naive_params)
    naive_file.close()
    diff_params = flax.serialization.msgpack_serialize(diff_params)
    diff_file = open("diff_params.msgpack", "wb")
    diff_file.write(diff_params)
    diff_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,  default="./configs/base.yaml",
                        help="yaml file for config.")
    args = parser.parse_args()

    main(args)
