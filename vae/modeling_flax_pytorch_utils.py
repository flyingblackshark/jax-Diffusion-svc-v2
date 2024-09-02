import re

import jax.numpy as jnp
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.random import PRNGKey


def rename_key(key):
    regex = r"\w+[.]\d+"
    pats = re.findall(regex, key)
    for pat in pats:
        key = key.replace(pat, "_".join(pat.split(".")))
    return key


#####################
# PyTorch => Flax #
#####################


# Adapted from https://github.com/huggingface/transformers/blob/c603c80f46881ae18b2ca50770ef65fa4033eacd/src/transformers/modeling_flax_pytorch_utils.py#L69
# and https://github.com/patil-suraj/stable-diffusion-jax/blob/main/stable_diffusion_jax/convert_diffusers_to_jax.py
def rename_key_and_reshape_tensor(pt_tuple_key, pt_tensor, random_flax_state_dict):
    """Rename PT weight names to corresponding Flax weight names and reshape tensor if necessary"""
    # conv norm or layer norm
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("scale",)

    # rename attention layers
    if len(pt_tuple_key) > 1:
        for rename_from, rename_to in (
            ("to_out_0", "proj_attn"),
            ("to_k", "key"),
            ("to_v", "value"),
            ("to_q", "query"),
        ):
            if pt_tuple_key[-2] == rename_from:
                weight_name = pt_tuple_key[-1]
                weight_name = "kernel" if weight_name == "weight" else weight_name
                renamed_pt_tuple_key = pt_tuple_key[:-2] + (rename_to, weight_name)
                if renamed_pt_tuple_key in random_flax_state_dict:
                    assert (
                        random_flax_state_dict[renamed_pt_tuple_key].shape
                        == pt_tensor.T.shape
                    )
                    return renamed_pt_tuple_key, pt_tensor.T

    if (
        any("norm" in str_ for str_ in pt_tuple_key)
        and (pt_tuple_key[-1] == "bias")
        and (pt_tuple_key[:-1] + ("bias",) not in random_flax_state_dict)
        and (pt_tuple_key[:-1] + ("scale",) in random_flax_state_dict)
    ):
        renamed_pt_tuple_key = pt_tuple_key[:-1] + ("scale",)
        return renamed_pt_tuple_key, pt_tensor
    elif (
        pt_tuple_key[-1] in ["weight", "gamma"]
        and pt_tuple_key[:-1] + ("scale",) in random_flax_state_dict
    ):
        renamed_pt_tuple_key = pt_tuple_key[:-1] + ("scale",)
        return renamed_pt_tuple_key, pt_tensor

    # embedding
    if (
        pt_tuple_key[-1] == "weight"
        and pt_tuple_key[:-1] + ("embedding",) in random_flax_state_dict
    ):
        pt_tuple_key = pt_tuple_key[:-1] + ("embedding",)
        return renamed_pt_tuple_key, pt_tensor

    # conv layer
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("kernel",)
    if pt_tuple_key[-1] == "weight" and pt_tensor.ndim == 4:
        pt_tensor = pt_tensor.transpose(2, 3, 1, 0)
        return renamed_pt_tuple_key, pt_tensor

    # linear layer
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("kernel",)
    if pt_tuple_key[-1] == "weight":
        pt_tensor = pt_tensor.T
        return renamed_pt_tuple_key, pt_tensor

    # old PyTorch layer norm weight
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("weight",)
    if pt_tuple_key[-1] == "gamma":
        return renamed_pt_tuple_key, pt_tensor

    # old PyTorch layer norm bias
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("bias",)
    if pt_tuple_key[-1] == "beta":
        return renamed_pt_tuple_key, pt_tensor

    return pt_tuple_key, pt_tensor


def convert_pytorch_state_dict_to_flax(pt_state_dict, flax_state_dict):

    random_flax_state_dict = flatten_dict(flax_state_dict)
    flax_state_dict = {}

    # Need to change some parameters name to match Flax names
    for pt_key, pt_tensor in pt_state_dict.items():
        renamed_pt_key = rename_key(pt_key)
        pt_tuple_key = tuple(renamed_pt_key.split("."))

        # Correctly rename weight parameters
        flax_key, flax_tensor = rename_key_and_reshape_tensor(
            pt_tuple_key, pt_tensor, random_flax_state_dict
        )

        if flax_key in random_flax_state_dict:
            if flax_tensor.shape != random_flax_state_dict[flax_key].shape:  # type: ignore
                raise ValueError(
                    f"PyTorch checkpoint seems to be incorrect. Weight {pt_key} was expected to be of shape "
                    f"{random_flax_state_dict[flax_key].shape}, but is {flax_tensor.shape}."  # type: ignore
                )

        # also add unexpected weight so that warning is thrown
        flax_state_dict[flax_key] = jnp.asarray(flax_tensor)

    return unflatten_dict(flax_state_dict)
