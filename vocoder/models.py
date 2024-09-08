from typing import Any
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
#import flax.traverse_util
LRELU_SLOPE = 0.1
#import torch
#from util import get_mel
#from omegaconf import OmegaConf

class ResBlock1(nn.Module):
    channels:int
    kernel_size:int=3
    dilation:tuple=(1, 3, 5)
    precision : jax.lax.Precision = jax.lax.Precision.HIGHEST
    def setup(self):
       
        self.convs1 =[
            nn.WeightNorm(nn.Conv(self.channels,[ self.kernel_size], 1, kernel_dilation=self.dilation[0],precision=self.precision)),
            nn.WeightNorm(nn.Conv( self.channels, [self.kernel_size], 1, kernel_dilation=self.dilation[1],precision=self.precision)),
            nn.WeightNorm(nn.Conv( self.channels, [self.kernel_size], 1, kernel_dilation=self.dilation[2],precision=self.precision))]
        self.convs2 = [
            nn.WeightNorm(nn.Conv( self.channels, [self.kernel_size], 1, kernel_dilation=1,precision=self.precision)),
            nn.WeightNorm(nn.Conv( self.channels, [self.kernel_size], 1, kernel_dilation=1,precision=self.precision)),
            nn.WeightNorm(nn.Conv(self.channels, [self.kernel_size], 1, kernel_dilation=1,precision=self.precision))
        ]
        self.num_layers = len(self.convs1) + len(self.convs2)
        
    def __call__(self, x,train=True):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = nn.leaky_relu(x,0.1)
            xt = c1(xt.transpose(0,2,1)).transpose(0,2,1)
            xt = nn.leaky_relu(xt,0.1)
            xt = c2(xt.transpose(0,2,1)).transpose(0,2,1)
            x = xt + x
        return x
class SineGen(nn.Module):
    samp_rate:int
    harmonic_num:int=0
    sine_amp:float=0.1
    noise_std:float=0.003
    voiced_threshold:int=0
    flag_for_pulse:bool=False
    def setup(self):
        self.dim = self.harmonic_num + 1
        self.sampling_rate = self.samp_rate

    def _f02uv(self, f0):
        uv = jnp.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _f02sine(self, f0,upp):
        rad = f0 / self.sampling_rate * jnp.arange(1, upp + 1)
        rad2 = jnp.fmod(rad[..., -1:] + 0.5, 1.0) - 0.5
        rad_acc = jnp.fmod(rad2.cumsum(axis=1),1.0)
        rad += jnp.pad(rad_acc[:, :-1, :], ((0, 0),(1, 0),(0,0)))
        rad = jnp.reshape(rad,(f0.shape[0], -1, 1))
        rad = jnp.multiply(rad, jnp.arange(1, self.dim + 1).reshape(1, 1, -1))
        rand_ini = jax.random.uniform(self.make_rng('rnorms'),shape=(1, 1, self.dim))
        rand_ini = rand_ini.at[..., 0].set(0)
        rad += rand_ini
        sines = jnp.sin(2 * jnp.pi * rad)
        return sines

    def __call__(self, f0,upp):
        f0 = jnp.expand_dims(f0,-1)
        sine_waves = self._f02sine(f0,upp) * self.sine_amp
        uv = (f0 > self.voiced_threshold).astype(jnp.float32)
        uv = jax.image.resize(uv, shape=(uv.shape[0],uv.shape[1]*upp,uv.shape[2]), method='nearest')
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * jax.random.normal(self.make_rng('rnorms'),sine_waves.shape)
        sine_waves = sine_waves * uv + noise
        return sine_waves


class SourceModuleHnNSF(nn.Module):
    sampling_rate:int=44100
    sine_amp:float=0.1
    add_noise_std:float=0.003
    voiced_threshod:int=0
    harmonic_num:int = 0
    def setup(self):
        self.l_sin_gen = SineGen(
            self.sampling_rate, self.harmonic_num, self.sine_amp, self.add_noise_std, self.voiced_threshod
        )

        self.l_linear = nn.Dense(1)
        #self.l_tanh = torch.nn.Tanh()

    def __call__(self, x,upp):
        sine_wavs = self.l_sin_gen(x, upp)
        sine_merge = nn.tanh(self.l_linear(sine_wavs))
        return sine_merge

class Generator(nn.Module):
    config : Any
    precision : jax.lax.Precision = jax.lax.Precision.HIGHEST
    def setup(self):
        self.num_kernels = len(self.config.resblock_kernel_sizes)
        self.num_upsamples = len(self.config.upsample_rates)
        self.conv_pre = nn.WeightNorm(nn.Conv(features=self.config.upsample_initial_channel, kernel_size=[7], strides=[1],precision=self.precision))
        self.scale_factor = np.prod(self.config.upsample_rates)
        self.m_source = SourceModuleHnNSF(sampling_rate=self.config.sampling_rate,harmonic_num=8)
        noise_convs = []
        ups = []
        for i, (u, k) in enumerate(zip(self.config.upsample_rates, self.config.upsample_kernel_sizes)):
            ups.append(
                    nn.WeightNorm(nn.ConvTranspose(
                        self.config.upsample_initial_channel // (2 ** (i + 1)),
                        (k,),
                        (u,),
                        transpose_kernel = True,precision=self.precision))
                )
            if i + 1 < len(self.config.upsample_rates):
                stride_f0 = int(np.prod(self.config.upsample_rates[i + 1:]))
                noise_convs.append(
                    nn.Conv(
                        features=self.config.upsample_initial_channel // (2 ** (i + 1)),
                        kernel_size=[stride_f0 * 2],
                        strides=[stride_f0],precision=self.precision
                    )
                )
            else:
                noise_convs.append(
                    nn.Conv(features=self.config.upsample_initial_channel //
                           (2 ** (i + 1)), kernel_size=[1],precision=self.precision)
                )

        resblocks = []
        for i in range(len(ups)):
            ch = self.config.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(self.config.resblock_kernel_sizes, self.config.resblock_dilation_sizes):
                resblocks.append(ResBlock1(ch, k, d,precision=self.precision))

        self.conv_post =  nn.WeightNorm(nn.Conv(features=1, kernel_size=[7], strides=1 , use_bias=False,precision=self.precision))
        self.cond = nn.Conv(self.config.upsample_initial_channel, 1,precision=self.precision)
        self.ups = ups
        self.noise_convs = noise_convs
        self.resblocks = resblocks
        self.upp = int(np.prod(self.config.upsample_rates))
    def __call__(self, x, f0,train=True):
        har_source = self.m_source(f0,self.upp).transpose(0,2,1)
        x = self.conv_pre(x.transpose(0,2,1)).transpose(0,2,1)
        for i in range(self.num_upsamples):
            x = nn.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x.transpose(0,2,1)).transpose(0,2,1)
            x_source = self.noise_convs[i](har_source.transpose(0,2,1)).transpose(0,2,1)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x,train=train)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x,train=train)
            x = xs / self.num_kernels
        x = nn.leaky_relu(x)
        x = self.conv_post(x.transpose(0,2,1)).transpose(0,2,1)
        x = nn.tanh(x) 
        return x

# if __name__ == "__main__":

#     packs = torch.load("lynx-combo-opencpop-kiritan-vocoder.ptc")
#     config = OmegaConf.load("./base.yaml")
#     model = Generator(config)
#     wav = jnp.ones((1,44100))
#     mel = get_mel(wav)
#     f0 = jnp.ones((1,100))
#     f0 = jax.image.resize(f0,shape=(f0.shape[0],mel.shape[-1]),method="nearest")
#     #n_frames = int(44100 // 512) + 1
#     #params = model.init(jax.random.PRNGKey(0),mel,f0)
#     #flatten_param = flax.traverse_util.flatten_dict(params,sep='.')
#     from convert import convert_torch_weights
#     params = convert_torch_weights("./lynx-combo-opencpop-kiritan-vocoder.ptc")
#     rng = {'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(0),'rnorms':jax.random.PRNGKey(0)}
#     model.apply({"params":params},mel,f0,rngs=rng)
#     breakpoint()