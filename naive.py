from jax import Array
import jax.numpy as jnp
import flax.linen as nn

from model_conformer_naive import ConformerNaiveEncoder
import numpy as np

f0_bin = 256
f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * np.log(1 + f0_min / 700)
f0_mel_max = 1127 * np.log(1 + f0_max / 700)


def f0_to_coarse(f0:Array) -> Array:
    f0_mel = 1127 * jnp.log(1 + f0 / 700)
    f0_mel = jnp.where(f0_mel>0,(f0_mel - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1,f0_mel)
    f0_mel = jnp.where(f0_mel <= 1 , 1 , f0_mel)
    f0_mel = jnp.where(f0_mel > (f0_bin - 1),f0_bin - 1,f0_mel)
    f0_coarse = jnp.rint(f0_mel).astype(jnp.int32)
    return f0_coarse

class Unit2MelNaive(nn.Module):
    #input_channel:int
    n_spk:int
    use_pitch_aug:bool=False
    out_dims:int=128
    use_speaker_encoder:bool=False
    speaker_encoder_out_channels:int=256
    n_chans:int = 256
    n_layers:int = 3
    expansion_factor:int = 2
    kernel_size:int = 31
    num_heads:int = 8
    use_norm:bool = False
    conv_only:bool = True
    conv_dropout:float = 0.
    atten_dropout:float = 0.1
    use_weight_norm:bool = False
    def setup(self):
        self.f0_embed = nn.Dense(self.n_chans)
        #self.volume_embed = nn.Dense(self.n_chans)
        if self.use_pitch_aug:
            self.aug_shift_embed = nn.Dense(self.n_chans, use_bias=False)
        else:
            self.aug_shift_embed = None

        if self.use_speaker_encoder:
            self.spk_embed = nn.Dense(self.n_chans, use_bias=False)
        else:
            if self.n_spk is not None and self.n_spk > 1:
                self.spk_embed = nn.Embed(self.n_spk, self.n_chans)

        # conv in stack
        self.ppg_stack = nn.Sequential([
            nn.Conv(self.n_chans, [3]),
            nn.GroupNorm(num_groups=4),
            nn.leaky_relu,
            nn.Conv(self.n_chans, [3])])
          
        # transformer
        self.decoder = ConformerNaiveEncoder(
            num_layers=self.n_layers,
            num_heads=self.num_heads,
            dim_model=self.n_chans,
            expansion_factor=self.expansion_factor,
            kernel_size=self.kernel_size,
            use_norm=self.use_norm,
            conv_only=self.conv_only,
            conv_dropout=self.conv_dropout,
            atten_dropout=self.atten_dropout,
            # conv_model_type=self.conv_model_type,
            # conv_model_activation=self.conv_model_activation
        )

        self.norm = nn.LayerNorm()
        # out
        self.dense_out = nn.Dense(self.out_dims)


    def __call__(self, ppg ,f0, spk_id=None, spk_mix_dict=None, aug_shift=None,
                gt_spec=None, train=True):

        '''
        input:
            B x n_frames x n_unit
        return:
            dict of B x n_frames x feat
        '''
        x = self.ppg_stack(ppg).transpose(0,2,1)
        #v = self.vec_stack(vec).transpose(0,2,1)
        f0 = jnp.expand_dims(f0,-1)
        x = x + self.f0_embed(jnp.log(1+ f0 / 700)).transpose(0,2,1) #+ self.volume_embed(volume).transpose(0,2,1)
        x = x.transpose(0,2,1)
        # if self.use_speaker_encoder:
        #     if spk_mix_dict is not None:
        #         assert spk_emb_dict is not None
        #         for k, v in spk_mix_dict.items():
        #             spk_id_torch = spk_emb_dict[str(k)]
        #             spk_id_torch = np.tile(spk_id_torch, (len(units), 1))
        #             spk_id_torch = torch.from_numpy(spk_id_torch).float().to(units.device)
        #             x = x + v * self.spk_embed(spk_id_torch)
        #     else:
        #         x = x + self.spk_embed(spk_emb)
        # else:
        #     if self.n_spk is not None and self.n_spk > 1:
        #         if spk_mix_dict is not None:
        #             for k, v in spk_mix_dict.items():
        #                 spk_id_torch = torch.LongTensor(np.array([[k]])).to(units.device)
        #                 x = x + v * self.spk_embed(spk_id_torch - 1)
        #         else:
        #             x = x + self.spk_embed(spk_id - 1)
        # if self.aug_shift_embed is not None and aug_shift is not None:
        #     x = x + self.aug_shift_embed(aug_shift / 5)
        x = self.decoder(x,train=train)
        x = self.norm(x)
        x = self.dense_out(x)
        return x

