import functools
from loguru import logger
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, cast, Dict
from input_pipeline.dataset import get_dataset
import fire
import jax
import jax.experimental.compilation_cache.compilation_cache
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from chex import PRNGKey
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import random
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.stages import Compiled, Wrapped
from PIL import Image
# from tensorboardX import SummaryWriter
# from tqdm import tqdm
from omegaconf import OmegaConf
from models import NaiveV2Diff
from naive import Unit2MelNaive
from sampling import rectified_flow_sample, rectified_flow_step
from step import naive_step
from profiling import memory_usage_params, trace_module_calls, get_peak_flops

jax.experimental.compilation_cache.compilation_cache.set_cache_dir("jit_cache")


print("JAX devices: ", jax.devices())
logger.info(f"JAX host count: {jax.device_count()}")


# def fmt_float_display(val: float | int) -> str:
#     if val > 1e3:
#         return f"{val:.2e}"
#     return f"{val:3.3f}"


class Trainer:

    def __init__(
        self,
        rng: PRNGKey,
        hp : Any,
        learning_rate: float = 5e-5,
        profile: bool = False,
        half_precision: bool = False,
    ) -> None:
        self.optimizer = optax.chain(
            optax.adam(learning_rate=learning_rate),
        )
        init_key, self.train_key = random.split(rng, 2)

        dtype = jnp.float16 if half_precision else jnp.float32

        self.diff_model = NaiveV2Diff(
            mel_channels=128,
            dim=512,
            use_mlp=True,
            mlp_factor=4,
            condition_dim=256,
            num_layers=20,
            expansion_factor=2,
            kernel_size=31,
            conv_only=True,
            use_norm=False,
            conv_dropout=0.0,
            atten_dropout=0.1,
        )
        self.naive_model = Unit2MelNaive(
                    n_spk=hp.model_naive.n_spk,
                    use_pitch_aug=hp.model_naive.use_pitch_aug,
                    out_dims=128,
                    use_speaker_encoder=False,
                    speaker_encoder_out_channels=256,
                    n_layers=hp.model_naive.n_layers,
                    n_chans=hp.model_naive.n_chans,
                    expansion_factor=hp.model_naive.expansion_factor,
                    kernel_size=hp.model_naive.kernel_size,
                    num_heads=hp.model_naive.num_heads,
                    use_norm=hp.model_naive.use_norm,
                    conv_only=hp.model_naive.conv_only,
                    conv_dropout=hp.model_naive.conv_dropout,
                    atten_dropout=hp.model_naive.atten_dropout,
                    use_weight_norm=hp.model_naive.use_weight_norm)
        
        n_devices = len(jax.devices())

        # x, y, t
        diff_input_values = (
            jnp.ones((n_devices, 400,128)),
            jnp.ones((n_devices, 400,128)),
            jnp.ones((n_devices), dtype=jnp.int32),
        )
        #ppg f0
        naive_input_values = (
        jnp.ones((n_devices,400,768)),
        jnp.ones((n_devices,400))
        )
        def create_diff_train_state(x, y, t, model, optimizer):
            variables = model.init(
                init_key,
                spec=x,
                diffusion_step=t,
                cond=y,
                rng=init_key,
                train=True,
            )
            train_state = TrainState.create(
                apply_fn=self.diff_model.apply, params=variables["params"], tx=optimizer
            )
            return train_state
        def create_naive_train_state(ppg, f0, model, optimizer):
            variables = model.init(
                init_key,
                ppg=ppg,
                f0=f0,
                #rng=init_key,
                train=True,
            )
            train_state = TrainState.create(
                apply_fn=self.naive_model.apply, params=variables["params"], tx=optimizer
            )
            return train_state
        logger.info(f"Available devices: {jax.devices()}")

        # Create a device mesh according to the physical layout of the devices.
        # device_mesh is just an ndarray
        device_mesh = mesh_utils.create_device_mesh((n_devices, 1))
        logger.info(f"Device mesh: {device_mesh}")

        # Async checkpointer for saving checkpoints across processes
        base_dir_abs = os.getcwd()
        options = ocp.CheckpointManagerOptions(max_to_keep=3)
        self.checkpoint_manager = ocp.CheckpointManager(
            f"{base_dir_abs}/checkpoints", options=options
        )

        # The axes are (data, model), so the mesh is (n_devices, 1) as the model is replicated across devices.
        # This object corresponds the axis names to the layout of the physical devices,
        # so that sharding a tensor along the axes shards according to the corresponding device_mesh layout.
        # i.e. with device layout of (8, 1), data would be replicated to all devices, and model would be replicated to 1 device.
        self.mesh = Mesh(device_mesh, axis_names=("data", "model"))

        logger.info(f"Mesh: {self.mesh}")

        self.data_iterator = get_dataset(hp,self.mesh)

        logger.info(f"Data Iterator: {self.data_iterator}")

        def get_sharding_for_spec(pspec: PartitionSpec) -> NamedSharding:
            """
            Get a NamedSharding for a given PartitionSpec, and the device mesh.
            A NamedSharding is simply a combination of a PartitionSpec and a Mesh instance.
            """
            return NamedSharding(self.mesh, pspec)

        # This shards the first dimension of the input data (batch dim) across the data axis of the mesh.
        x_sharding = get_sharding_for_spec(PartitionSpec("data"))

        # Returns a pytree of shapes for the train state
        diff_train_state_sharding_shape = jax.eval_shape(
            functools.partial(
                create_diff_train_state, model=self.diff_model, optimizer=self.optimizer
            ),
            *diff_input_values,
        )
        naive_train_state_sharding_shape = jax.eval_shape(
            functools.partial(
                create_naive_train_state, model=self.naive_model, optimizer=self.optimizer
            ),
            *naive_input_values,
        )

        # Get the PartitionSpec for all the variables in the train state
        diff_train_state_sharding = nn.get_sharding(diff_train_state_sharding_shape, self.mesh)
        naive_train_state_sharding = nn.get_sharding(naive_train_state_sharding_shape, self.mesh)
        diff_input_sharding: Any = (x_sharding, x_sharding, x_sharding)
        naive_input_sharding: Any = (x_sharding, x_sharding)

        logger.info(f"Initializing model...")
        # Shard the train_state so so that it's replicated across devices
        jit_create_diff_train_state_fn = jax.jit(
            create_diff_train_state,
            static_argnums=(3, 4),
            in_shardings=diff_input_sharding,  # type: ignore
            out_shardings=diff_train_state_sharding,
        )
        jit_create_naive_train_state_fn = jax.jit(
            create_naive_train_state,
            static_argnums=(2,3),
            in_shardings=naive_input_sharding,  # type: ignore
            out_shardings=naive_train_state_sharding,
        )
        self.diff_train_state = jit_create_diff_train_state_fn(
            *diff_input_values, self.diff_model, self.optimizer
        )
        self.naive_train_state = jit_create_naive_train_state_fn(
            *naive_input_values, self.naive_model, self.optimizer
        )
        diff_total_bytes, diff_total_params = memory_usage_params(self.diff_train_state.params)
        naive_total_bytes, naive_total_params = memory_usage_params(self.naive_train_state.params)
        logger.info(f"Diff Model parameter count: {diff_total_params} using: {diff_total_bytes}")
        logger.info(f"Naive Model parameter count: {naive_total_params} using: {naive_total_bytes}")

        logger.info("JIT compiling step functions...")

        diff_step_in_sharding: Any = (
            diff_train_state_sharding,
            x_sharding,
            x_sharding,
            x_sharding,
        )
        naive_step_in_sharding: Any = (
            naive_train_state_sharding,
            x_sharding,
            x_sharding,
            x_sharding,
            x_sharding,
        )
        diff_step_out_sharding: Any = (
            get_sharding_for_spec(PartitionSpec()),
            diff_train_state_sharding,
        )
        naive_step_out_sharding: Any = (
            get_sharding_for_spec(PartitionSpec()),
            naive_train_state_sharding,
            get_sharding_for_spec(PartitionSpec()),
        )
        self.diff_train_step: Wrapped = jax.jit(
            functools.partial(rectified_flow_step, training=True),
            in_shardings=diff_step_in_sharding,
            out_shardings=diff_step_out_sharding,
        )
        self.naive_train_step: Wrapped = jax.jit(
            functools.partial(naive_step, training=True),
            in_shardings=naive_step_in_sharding,
            out_shardings=naive_step_out_sharding,
        )
        self.diff_eval_step: Wrapped = jax.jit(
            functools.partial(rectified_flow_step, training=False),
            in_shardings=diff_step_in_sharding,
            out_shardings=diff_step_out_sharding,
        )

        if profile:
            logger.info("AOT compiling step functions...")
            compiled_step: Compiled = self.diff_train_step.lower(
                self.diff_train_state, *diff_input_values[:2], init_key
            ).compile()
            train_cost_analysis: Dict = compiled_step.cost_analysis()[0]  # type: ignore
            self.flops_for_step = train_cost_analysis.get("flops", 0)
            logger.info(
                f"Steps compiled, train cost analysis FLOPs: {self.flops_for_step}"
            )
        else:
            self.flops_for_step = 0

    def save_checkpoint(self, global_step: int):
        if self.diff_train_state is not None:
            self.checkpoint_manager.save(global_step, args=ocp.args.StandardSave(self.diff_train_state))  # type: ignore


# def run_eval(
#     eval_dataset: Dataset,
#     n_eval_batches: int,
#     dataset_config: DatasetConfig,
#     trainer: Trainer,
#     rng: PRNGKey,
#     summary_writer: SummaryWriter,
#     iter_description_dict: Dict,
#     global_step: int,
#     do_sample: bool,
#     epoch: int,
# ):
#     """
#     Run evaluation on the eval subset, and optionally sample the model
#     """
#     num_eval_batches = 1
#     eval_iter = tqdm(
#         eval_dataset.iter(batch_size=16, drop_last_batch=True),
#         leave=False,
#         total=num_eval_batches,
#         dynamic_ncols=True,
#     )
#     for j, eval_batch in enumerate(eval_iter):
#         if j >= n_eval_batches:
#             break

#         # Eval loss
#         images, labels = process_batch(
#             eval_batch,
#             dataset_config.latent_size,
#             dataset_config.n_channels,
#             dataset_config.label_field_name,
#             dataset_config.image_field_name,
#         )
#         eval_loss, trainer.train_state = trainer.eval_step(
#             trainer.train_state, images, labels, rng
#         )
#         iter_description_dict.update({"eval_loss": fmt_float_display(eval_loss)})
#         eval_iter.set_postfix(iter_description_dict)
#         summary_writer.add_scalar("eval_loss", eval_loss, global_step)

#         # Sampling
#         if do_sample:
#             sample_key, rng = random.split(rng)
#             n_labels_to_sample = (
#                 dataset_config.n_labels_to_sample
#                 if dataset_config.n_labels_to_sample
#                 else dataset_config.n_classes
#             )
#             noise_shape = (
#                 n_labels_to_sample,
#                 dataset_config.n_channels,
#                 dataset_config.latent_size,
#                 dataset_config.latent_size,
#             )
#             init_noise = random.normal(rng, noise_shape)
#             labels = jnp.arange(0, n_labels_to_sample)
#             null_cond = jnp.ones_like(labels) * 10
#             samples = rectified_flow_sample(
#                 trainer.train_state,
#                 init_noise,
#                 labels,
#                 sample_key,
#                 null_cond=null_cond,
#                 sample_steps=50,
#             )
#             grid = image_grid(samples)
#             sample_img_filename = f"samples/epoch_{epoch}_globalstep_{global_step}.png"
#             grid.save(sample_img_filename)


def main(
    n_epochs: int = 100,
    learning_rate: float = 1e-4,
    eval_save_steps: int = 250,
    n_eval_batches: int = 1,
    sample_every_n: int = 1,
    dataset_name: str = "imagenet",
    profile: bool = False,
    half_precision: bool = False,
    **kwargs,
):
    """
    Arguments:
        n_epochs: Number of epochs to train for.
        batch_size: Batch size for training.
        learning_rate: Learning rate for the optimizer.
        eval_save_steps: Number of steps between evaluation runs and checkpoint saves.
        n_eval_batches: Number of batches to evaluate on.
        sample_every_n: Number of epochs between sampling runs.
        dataset_name: Name of the dataset config to select, valid options are in DATASET_CONFIGS.
        profile: Run a single train and eval step, and print out the cost analysis, then exit.
        half_precision: case the model to fp16 for training.
    """
    hp = OmegaConf.load("configs/base.yaml")
    rng = random.PRNGKey(hp.train.seed)
    trainer = Trainer(rng, hp, learning_rate, profile, half_precision)

    #summary_writer = SummaryWriter(flush_secs=1, max_queue=1)
    #ensure_directory("samples", clear=True)

    #iter_description_dict = {"loss": 0.0, "eval_loss": 0.0, "epoch": 0, "step": 0}
    #n_samples = dataset_config.dataset_length or len(train_dataset)

    n_evals = 0
    init_epoch = 0
    example_batch = None
    for step in range(init_epoch, hp.train.total_steps):
        example_batch = next(trainer.data_iterator)
        
        # Train step
        step_key = random.PRNGKey(step)

        if profile:
            # profile_ctx = jax.profiler.trace(
            #     profiler_trace_dir, create_perfetto_link=True
            # )
            profile_ctx = nullcontext()
        else:
            profile_ctx = nullcontext()

        with profile_ctx:
            #step_start_time = time.perf_counter()
            naive_train_loss, naive_updated_state,cond_mel = trainer.naive_train_step(
                trainer.naive_train_state, 
                example_batch['hubert_feature'], 
                example_batch['f0_feature'],
                example_batch['mel_feature'], 
                step_key
            )
            trainer.naive_train_state = naive_updated_state

            diff_train_loss, diff_updated_state = trainer.diff_train_step(
                trainer.diff_train_state, 
                example_batch['mel_feature'], 
                cond_mel, 
                step_key
            )

            trainer.diff_train_state = diff_updated_state
            

            logger.info(f"naive_train_loss : {naive_train_loss} diff_train_loss : {diff_train_loss}")
            # step_duration = time.perf_counter() - step_start_time
            # flops_device_sec = trainer.flops_for_step / (
            #     step_duration * device_count
            # )

            #peak_flops = get_peak_flops()
            #mfu = flops_device_sec / peak_flops

        #     summary_writer.add_scalar(
        #         "train_step_time",
        #         step_duration,
        #         global_step,
        #     )

        # summary_writer.add_scalar("train_loss", train_loss, global_step)

        # if global_step % eval_save_steps == 0 or profile:
        #     trainer.save_checkpoint(global_step)
        #     run_eval(
        #         eval_dataset,
        #         n_eval_batches,
        #         dataset_config,
        #         trainer,
        #         rng,
        #         summary_writer,
        #         iter_description_dict,
        #         global_step,
        #         n_evals % sample_every_n == 0,
        #         epoch,
        #     )

        if profile:
            logger.info("\nExiting after profiling a single step.")
            return

if __name__ == "__main__":
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    fire.Fire(main)
