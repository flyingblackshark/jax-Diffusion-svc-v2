# coding=utf-8
# Copyright 2021 The Fairseq Authors and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Flax Hubert model."""

from functools import partial
from typing import Optional, Tuple, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from transformers import HubertConfig
from transformers.modeling_flax_outputs import FlaxBaseModelOutput
from transformers.modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
)
from transformers.utils import ModelOutput, logging

logger = logging.get_logger(__name__)


@flax.struct.dataclass
class FlaxHubertOutput(ModelOutput):
    last_hidden_state: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None
    extract_features: jnp.ndarray = None


class FlaxConvWithWeightNorm(nn.Module):
    config: HubertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv = nn.Conv(
            features=self.config.hidden_size,
            kernel_size=(self.config.num_conv_pos_embeddings,),
            kernel_init=jax.nn.initializers.he_normal(),
            padding="VALID",
            feature_group_count=self.config.num_conv_pos_embedding_groups,
            dtype=self.dtype,
        )
        weight_shape = (
            self.conv.features,
            self.conv.features // self.conv.feature_group_count,
            self.conv.kernel_size[0],
        )
        self.weight_v = self.param(
            "weight_v", jax.nn.initializers.he_normal(), weight_shape
        )
        self.weight_g = self.param(
            "weight_g",
            lambda _: jnp.linalg.norm(self.weight_v, axis=(0, 1))[None, None, :],
        )
        self.bias = self.param("bias", jax.nn.initializers.zeros, (self.conv.features,))
        self.prev_padding = self.conv.kernel_size[0] // 2

    def _get_normed_weights(self):
        weight_v_norm = jnp.linalg.norm(self.weight_v, axis=(0, 1))[None, None, :]
        normed_weight_v = jnp.divide(self.weight_v, weight_v_norm)
        normed_kernel = jnp.multiply(normed_weight_v, self.weight_g)
        return normed_kernel

    def __call__(self, hidden_states):
        kernel = self._get_normed_weights()
        hidden_states = jnp.pad(
            hidden_states, ((0, 0), (self.prev_padding, self.prev_padding), (0, 0))
        )
        hidden_states = self.conv.apply(
            {"params": {"kernel": kernel.T, "bias": self.bias}}, hidden_states
        )
        return hidden_states


class FlaxHubertNoLayerNormConvLayer(nn.Module):
    config: HubertConfig
    layer_id: int = 0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.in_conv_dim = (
            self.config.conv_dim[self.layer_id - 1] if self.layer_id > 0 else 1
        )
        self.out_conv_dim = self.config.conv_dim[self.layer_id]

        self.conv = nn.Conv(
            features=self.config.conv_dim[self.layer_id],
            kernel_size=(self.config.conv_kernel[self.layer_id],),
            strides=(self.config.conv_stride[self.layer_id],),
            use_bias=self.config.conv_bias,
            kernel_init=jax.nn.initializers.he_normal(),
            padding="VALID",
            dtype=self.dtype,
        )
        self.activation = ACT2FN[self.config.feat_extract_activation]

    def __call__(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class FlaxHubertLayerNormConvLayer(nn.Module):
    config: HubertConfig
    layer_id: int = 0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.in_conv_dim = (
            self.config.conv_dim[self.layer_id - 1] if self.layer_id > 0 else 1
        )
        self.out_conv_dim = self.config.conv_dim[self.layer_id]

        self.conv = nn.Conv(
            features=self.config.conv_dim[self.layer_id],
            kernel_size=(self.config.conv_kernel[self.layer_id],),
            strides=(self.config.conv_stride[self.layer_id],),
            use_bias=self.config.conv_bias,
            kernel_init=jax.nn.initializers.he_normal(),
            padding="VALID",
            dtype=self.dtype,
        )
        self.layer_norm = nn.LayerNorm(
            epsilon=self.config.layer_norm_eps, dtype=self.dtype
        )
        self.activation = ACT2FN[self.config.feat_extract_activation]

    def __call__(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class FlaxHubertGroupNormConvLayer(nn.Module):
    config: HubertConfig
    layer_id: int = 0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.in_conv_dim = (
            self.config.conv_dim[self.layer_id - 1] if self.layer_id > 0 else 1
        )
        self.out_conv_dim = self.config.conv_dim[self.layer_id]

        self.conv = nn.Conv(
            features=self.config.conv_dim[self.layer_id],
            kernel_size=(self.config.conv_kernel[self.layer_id],),
            strides=(self.config.conv_stride[self.layer_id],),
            use_bias=self.config.conv_bias,
            kernel_init=jax.nn.initializers.he_normal(),
            padding="VALID",
            dtype=self.dtype,
        )
        self.activation = ACT2FN[self.config.feat_extract_activation]

        self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim, dtype=self.dtype,epsilon=1e-5)

    def __call__(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class FlaxHubertPositionalConvEmbedding(nn.Module):
    config: HubertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv = FlaxConvWithWeightNorm(self.config, dtype=self.dtype)
        self.activation = ACT2FN[self.config.feat_extract_activation]
        self.num_pad_remove = 1 if self.config.num_conv_pos_embeddings % 2 == 0 else 0

    def __call__(self, hidden_states):
        hidden_states = hidden_states.transpose((0, 1, 2))

        hidden_states = self.conv(hidden_states)

        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, : -self.num_pad_remove, :]
        hidden_states = self.activation(hidden_states)

        hidden_states = hidden_states.transpose((0, 1, 2))
        return hidden_states


class FlaxConvLayersCollection(nn.Module):
    config: HubertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if self.config.feat_extract_norm == "layer":
            self.layers = [
                FlaxHubertLayerNormConvLayer(
                    self.config, layer_id=i, name=str(i), dtype=self.dtype
                )
                for i in range(self.config.num_feat_extract_layers)
            ]
        elif self.config.feat_extract_norm == "group":
            self.layers = [
                FlaxHubertGroupNormConvLayer(
                    self.config, layer_id=0, name=str(0), dtype=self.dtype
                )] + [
                FlaxHubertNoLayerNormConvLayer(
                    self.config, layer_id=i, name=str(i), dtype=self.dtype
                )
                for i in range(1, self.config.num_feat_extract_layers)
            ]
        else:
            raise ValueError(
                f"`config.feat_extract_norm` is {self.config.feat_extract_norm}, but has to be one of ['group',"
                " 'layer']"
            )

    def __call__(self, hidden_states):
        for i, conv_layer in enumerate(self.layers):
            hidden_states = conv_layer(hidden_states)
        return hidden_states


class FlaxHubertFeatureEncoder(nn.Module):
    config: HubertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv_layers = FlaxConvLayersCollection(self.config, dtype=self.dtype)

    def __call__(self, input_values, freeze_feature_encoder=False):
        hidden_states = input_values[:, :, None]
        hidden_states = self.conv_layers(hidden_states)
        if freeze_feature_encoder:
            hidden_states = jax.lax.stop_gradient(hidden_states)
        return hidden_states


class FlaxHubertFeatureProjection(nn.Module):
    config: HubertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.feat_proj_layer_norm = self.config.feat_proj_layer_norm
        if self.feat_proj_layer_norm:
            self.layer_norm = nn.LayerNorm(
                epsilon=self.config.layer_norm_eps, dtype=self.dtype
            )
        self.projection = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.feat_proj_dropout)

    def __call__(self, hidden_states, deterministic=True):
        if self.feat_proj_layer_norm:
            hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states


class FlaxHubertAttention(nn.Module):
    config: HubertConfig
    embed_dim: int
    num_heads: int
    dropout: float = 0.0
    bias: bool = True
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self) -> None:
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        dense = partial(
            nn.Dense,
            self.embed_dim,
            use_bias=self.bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

        self.q_proj, self.k_proj, self.v_proj = dense(), dense(), dense()
        self.out_proj = dense()

        self.dropout_layer = nn.Dropout(rate=self.dropout)

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(
            hidden_states.shape[:2] + (self.num_heads, self.head_dim)
        )

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        output_attentions: bool = False,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """Input shape: Batch x Time x Channel"""

        # get query, key, value proj for self_attention
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self._split_heads(query_states)
        key_states = self._split_heads(key_states)
        value_states = self._split_heads(value_states)

        if attention_mask is not None:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(
                    self.dtype
                ),
            )
        else:
            attention_bias = None

        dropout_rng = None
        if not deterministic and self.dropout > 0.0:
            dropout_rng = self.make_rng("dropout")

        attn_weights = dot_product_attention_weights(
            query_states,
            key_states,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout,
            broadcast_dropout=True,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class FlaxHubertFeedForward(nn.Module):
    config: HubertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.intermediate_dropout = nn.Dropout(self.config.activation_dropout)

        self.intermediate_dense = nn.Dense(
            self.config.intermediate_size, dtype=self.dtype
        )
        if isinstance(self.config.hidden_act, str):
            self.intermediate_activation = ACT2FN[self.config.hidden_act]
        else:
            self.intermediate_activation = self.config.hidden_act

        self.output_dense = nn.Dense(self.config.hidden_size, dtype=self.dtype)
        self.output_dropout = nn.Dropout(self.config.activation_dropout)

    def __call__(self, hidden_states, deterministic=True):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_activation(hidden_states)
        hidden_states = self.intermediate_dropout(
            hidden_states, deterministic=deterministic
        )

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states, deterministic=deterministic)

        return hidden_states


class FlaxHubertEncoderLayer(nn.Module):
    config: HubertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.attention = FlaxHubertAttention(
            config=self.config,
            embed_dim=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            dropout=self.config.attention_dropout,
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(self.config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(
            epsilon=self.config.layer_norm_eps, dtype=self.dtype
        )
        self.feed_forward = FlaxHubertFeedForward(self.config, dtype=self.dtype)
        self.final_layer_norm = nn.LayerNorm(
            epsilon=self.config.layer_norm_eps, dtype=self.dtype
        )

    def __call__(
        self,
        hidden_states,
        attention_mask: Optional[jnp.ndarray] = None,
        output_attentions: bool = False,
        deterministic=True,
    ):
        attn_residual = hidden_states
        hidden_states, attn_weights = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            deterministic=deterministic,
        )

        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = attn_residual + hidden_states

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(
            hidden_states, deterministic=deterministic
        )
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class FlaxHubertEncoderLayerStableLayerNorm(nn.Module):
    config: HubertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.attention = FlaxHubertAttention(
            config=self.config,
            embed_dim=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            dropout=self.config.attention_dropout,
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(self.config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(
            epsilon=self.config.layer_norm_eps, dtype=self.dtype
        )
        self.feed_forward = FlaxHubertFeedForward(self.config, dtype=self.dtype)
        self.final_layer_norm = nn.LayerNorm(
            epsilon=self.config.layer_norm_eps, dtype=self.dtype
        )

    def __call__(
        self,
        hidden_states,
        attention_mask: Optional[jnp.ndarray] = None,
        output_attentions: bool = False,
        deterministic=True,
    ):
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            deterministic=deterministic,
        )

        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = attn_residual + hidden_states

        hidden_states = hidden_states + self.feed_forward(
            self.final_layer_norm(hidden_states), deterministic=deterministic
        )

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class FlaxHubertLayerCollection(nn.Module):
    config: HubertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layers = [
            FlaxHubertEncoderLayer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(
                hidden_states,
                attention_mask,
                deterministic=deterministic,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = (hidden_states, all_hidden_states, all_attentions)

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class FlaxHubertEncoder(nn.Module):
    config: HubertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.pos_conv_embed = FlaxHubertPositionalConvEmbedding(
            self.config, dtype=self.dtype
        )
        self.layer_norm = nn.LayerNorm(
            epsilon=self.config.layer_norm_eps, dtype=self.dtype
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout)
        self.layers = FlaxHubertLayerCollection(self.config, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        attention_mask: Optional[jnp.ndarray] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ):
        if attention_mask is not None:
            # make sure padded tokens are not attended to
            hidden_states = jnp.where(
                jnp.broadcast_to(attention_mask[:, :, None], hidden_states.shape),
                hidden_states,
                0,
            )

        position_embeddings = self.pos_conv_embed(hidden_states)

        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)

        outputs = self.layers(
            hidden_states,
            attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        #last_hidden_state = self.layer_norm(outputs[0])
        last_hidden_state = outputs[0]


        hidden_states = None
        if output_hidden_states:
            hidden_states = outputs[1]
            hidden_states = hidden_states[:-1] + (last_hidden_state,)

        if not return_dict:
            outputs = (last_hidden_state, hidden_states) + (
                outputs[2:] if output_hidden_states else outputs[1:]
            )
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
        )


class FlaxHubertLayerStableLayerNormCollection(nn.Module):
    config: HubertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layers = [
            FlaxHubertEncoderLayerStableLayerNorm(
                self.config, name=str(i), dtype=self.dtype
            )
            for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(
                hidden_states,
                attention_mask,
                deterministic=deterministic,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = (hidden_states, all_hidden_states, all_attentions)

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class FlaxHubertEncoderStableLayerNorm(nn.Module):
    config: HubertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.pos_conv_embed = FlaxHubertPositionalConvEmbedding(
            self.config, dtype=self.dtype
        )
        self.layer_norm = nn.LayerNorm(
            epsilon=self.config.layer_norm_eps, dtype=self.dtype
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout)
        self.layers = FlaxHubertLayerStableLayerNormCollection(
            self.config, dtype=self.dtype
        )

    def __call__(
        self,
        hidden_states,
        attention_mask: Optional[jnp.ndarray] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ):
        if attention_mask is not None:
            hidden_states = jnp.where(
                jnp.broadcast_to(attention_mask[:, :, None], hidden_states.shape),
                hidden_states,
                0,
            )

        position_embeddings = self.pos_conv_embed(hidden_states)

        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)

        outputs = self.layers(
            hidden_states,
            attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        #last_hidden_state = self.layer_norm(outputs[0])
        last_hidden_state = outputs[0]

        hidden_states = None
        if output_hidden_states:
            hidden_states = outputs[1]
            hidden_states = hidden_states[:-1] + (last_hidden_state,)

        if not return_dict:
            outputs = (last_hidden_state, hidden_states) + (
                outputs[2:] if output_hidden_states else outputs[1:]
            )
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
        )


class FlaxHubertPreTrainedModel(FlaxPreTrainedModel):
    config_class = HubertConfig
    base_model_prefix = "hubert"
    main_input_name = "input_values"
    module_class: nn.Module = None
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(
        self,
        config: HubertConfig,
        input_shape: Tuple = (1, 1024),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(
            config,
            module,
            input_shape=input_shape,
            seed=seed,
            dtype=dtype,
            _do_init=_do_init,
        )

    def init_weights(
        self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None
    ) -> FrozenDict:
        input_values = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_values)
        params_rng, dropout_rng = jax.random.split(rng, 2)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        random_params = self.module.init(
            rngs, input_values, attention_mask, return_dict=False
        )["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def __call__(
        self,
        input_values,
        attention_mask=None,
        mask_time_indices=None,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        freeze_feature_encoder: bool = False,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        batch_size, sequence_length = input_values.shape

        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params}

        return self.module.apply(
            inputs,
            jnp.array(input_values, dtype="f4"),
            jnp.array(attention_mask, dtype="i4"),
            mask_time_indices,
            not train,
            output_attentions,
            output_hidden_states,
            freeze_feature_encoder,
            return_dict,
            rngs=rngs,
        )


class FlaxHubertModule(nn.Module):
    config: HubertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.feature_extractor = FlaxHubertFeatureEncoder(self.config, dtype=self.dtype)
        self.feature_projection = FlaxHubertFeatureProjection(
            self.config, dtype=self.dtype
        )

        if self.config.mask_time_prob > 0.0 or self.config.mask_feature_prob > 0.0:
            self.masked_spec_embed = self.param(
                "masked_spec_embed",
                nn.initializers.uniform(dtype=self.dtype),
                (self.config.hidden_size,),
            )

        if self.config.do_stable_layer_norm:
            self.encoder = FlaxHubertEncoderStableLayerNorm(self.config)
        else:
            self.encoder = FlaxHubertEncoder(self.config)

    def __call__(
        self,
        input_values: Optional[jnp.ndarray],
        attention_mask: Optional[jnp.ndarray] = None,
        mask_time_indices: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        freeze_feature_encoder: bool = False,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, FlaxHubertOutput]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        extract_features = self.feature_extractor(input_values, freeze_feature_encoder)

        if attention_mask is not None:
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask
            )

        hidden_states = self.feature_projection(
            extract_features, deterministic=deterministic
        )
        if mask_time_indices is not None:
            hidden_states = jnp.where(
                jnp.broadcast_to(mask_time_indices[:, :, None], hidden_states.shape),
                jnp.broadcast_to(
                    self.masked_spec_embed[None, None, :], hidden_states.shape
                ),
                hidden_states,
            )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return FlaxHubertOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            extract_features=extract_features,
        )

    def _get_feat_extract_output_lengths(self, input_lengths: Union[jnp.ndarray, int]):
        def _conv_out_length(input_length, kernel_size, stride):
            return (input_length - kernel_size) // stride + 1

        for kernel_size, stride in zip(
            self.config.conv_kernel, self.config.conv_stride
        ):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths

    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: jnp.ndarray
    ):
        non_padded_lengths = attention_mask.cumsum(axis=-1)[:, -1]

        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths)

        batch_size = attention_mask.shape[0]

        attention_mask = jnp.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype
        )
        attention_mask = attention_mask.at[
            jnp.arange(attention_mask.shape[0]), output_lengths - 1
        ].set(1)
        attention_mask = jnp.flip(jnp.flip(attention_mask, -1).cumsum(-1), -1).astype(
            "bool"
        )
        return attention_mask


class FlaxHubertModel(FlaxHubertPreTrainedModel):
    module_class = FlaxHubertModule
