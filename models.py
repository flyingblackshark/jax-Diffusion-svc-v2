from chex import PRNGKey
from jax import Array, lax
import jax.numpy as jnp
import flax.linen as nn
import jax
from jax.nn import initializers, swish
from typing import Optional
from flax.linen import LayerNorm, dot_product_attention, dot_product_attention_weights
from jax.lax import Precision

from model_conformer_naive import ConformerConvModule

class TimestepEmbedder(nn.Module):
    """
    Rotational positional encoding for tiemestep.
    TODO can we dedupe the positional encoding code?
    """

    hidden_size: int
    frequency_embedding_size: int

    @nn.compact
    def __call__(self, t: Array) -> Array:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb: Array = nn.Sequential(
            [
                nn.Dense(self.hidden_size),
                nn.silu,
                nn.Dense(self.hidden_size),
            ]
        )(t_freq)
        return t_emb

    @staticmethod
    def timestep_embedding(t: Array, freq_emb_size: int, max_period=10000):
        """
        Apply RoPE to timestep.
        """
        half = freq_emb_size // 2
        freqs = jnp.exp(-jnp.log(max_period) * jnp.arange(0, half) / half)
        args = t[:, None] * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if freq_emb_size % 2:
            embedding = jnp.concatenate(
                [embedding, jnp.zeros_like(embedding[:, :1])], axis=-1
            )
        return embedding
class NaiveV2DiffLayer(nn.Module):
    dim_model: int
    dim_cond: int
    conv_dropout: float = 0.
    atten_dropout: float = 0.1
    use_mlp=True
    expansion_factor=2
    kernel_size=31
    def setup(self):

        self.conformer = ConformerConvModule(
            self.dim_model,
            expansion_factor=self.expansion_factor,
            kernel_size=self.kernel_size,
            dropout=self.conv_dropout,
        )

        self.diffusion_step_projection = nn.Conv(self.dim_model, 1)
        self.condition_projection = nn.Conv(self.dim_model, 1)

    def __call__(self, x, condition=None, diffusion_step=None):
        res_x = x
        x = x + self.diffusion_step_projection(diffusion_step) + self.condition_projection(condition)
        x = self.conformer(x)  # (#batch, dim_model, length)
        x = x + res_x
        return x  # (#batch, length, dim_model)
class NaiveV2Diff(nn.Module):
    mel_channels:int=128
    dim:int=512
    use_mlp:bool=True
    mlp_factor:int=4
    num_layers:int=20
    expansion_factor:int=2
    kernel_size:int=31
    conv_dropout:float=0.0
    atten_dropout:float=0.1
    def setup(self):
        self.input_projection = nn.Conv(self.dim, [1])
        self.diffusion_embedding = TimestepEmbedder(min(self.dim, 1024), 256)
        self.conditioner_projection = nn.Sequential([
            nn.Conv(self.dim * self.mlp_factor, 1),
            nn.gelu,
            nn.Conv(self.dim, 1)]
        )

        self.residual_layers = [
                NaiveV2DiffLayer(
                    dim_model=self.dim,
                    dim_cond=self.dim,
                    conv_dropout=self.conv_dropout,
                    atten_dropout=self.atten_dropout
                )
                for i in range(self.num_layers)
            ]
        
        
        self.output_projection = nn.Sequential([
            nn.Conv(self.dim * self.mlp_factor, kernel_size=1),
            nn.gelu,
            nn.Conv(self.mel_channels, kernel_size=1,kernel_init=initializers.zeros)]
        )

    def __call__(self, spec, diffusion_step, cond,rng,train=True):
        x = spec
        conditioner = cond
        """
        :param x: [B, M, T]
        :param diffusion_step: [B,]
        :param conditioner: [B, M, T]
        :return:
        """

        x = self.input_projection(x)  # x [B, residual_channel, T]
        x = nn.gelu(x)

        diffusion_step = jnp.expand_dims(self.diffusion_embedding(diffusion_step),1)
        condition = self.conditioner_projection(conditioner)

        for layer in self.residual_layers:
            x = layer(x, condition, diffusion_step)

        # MLP and GLU
        x = self.output_projection(x)  # [B, 128, T]

        return x
