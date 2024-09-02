
import jax
from jax import Array, random
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax
def naive_step(naive_state: TrainState, 
                 ppg : Array , 
                 pit : Array,
                 spec : Array,
                 prng_key:Array,
                 training: bool = True):
    prng_key, step_key = random.split(prng_key)
    def naive_loss(params):
        fake_mel = naive_state.apply_fn({'params': params},ppg=ppg,f0=pit,train=training,rngs={'rnorms':step_key})
        loss_naive = optax.squared_error(fake_mel, spec)
        return loss_naive.mean(),fake_mel

    (loss,fake_mel), grads = jax.value_and_grad(naive_loss,has_aux=True)(naive_state.params)
    return loss,naive_state.apply_gradients(grads=grads),fake_mel