
import jax.numpy as jnp
import flax.linen as nn
import jax
from model_conformer_naive import ConformerNaiveEncoder

class Unit2MelNaive(nn.Module):
    n_spk:int
    out_dims:int=128
    use_speaker_encoder:bool=False
    speaker_encoder_out_channels:int=256
    n_chans:int = 256
    n_layers:int = 3
    expansion_factor:int = 2
    kernel_size:int = 31
    num_heads:int = 8
    conv_dropout:float = 0.
    atten_dropout:float = 0.1
    precision : jax.lax.Precision = jax.lax.Precision.HIGHEST
    def setup(self):
        self.f0_embed = nn.Dense(self.n_chans)
        self.volume_embed = nn.Dense(self.n_chans)

        if self.use_speaker_encoder:
            self.spk_embed = nn.Dense(self.n_chans, use_bias=False)
        else:
            if self.n_spk is not None and self.n_spk > 1:
                self.spk_embed = nn.Embed(self.n_spk, self.n_chans)

        # conv in stack
        self.ppg_stack = nn.Sequential([
            nn.Conv(self.n_chans, [3],precision=self.precision),
            nn.GroupNorm(num_groups=4),
            nn.leaky_relu,
            nn.Conv(self.n_chans, [3],precision=self.precision)])
          
        # transformer
        self.decoder = ConformerNaiveEncoder(
            num_layers=self.n_layers,
            num_heads=self.num_heads,
            dim_model=self.n_chans,
            expansion_factor=self.expansion_factor,
            kernel_size=self.kernel_size,
            conv_dropout=self.conv_dropout,
            atten_dropout=self.atten_dropout,
            precision=self.precision
        )

        self.norm = nn.LayerNorm()
        # out
        self.dense_out = nn.WeightNorm(nn.Dense(self.out_dims,precision=self.precision))


    def __call__(self, ppg, f0, vol, spk_id=None, spk_mix_dict=None, t_spec=None, train=True):

        '''
        input:
            B x n_frames x n_unit
        return:
            dict of B x n_frames x feat
        '''
        f0 = jnp.expand_dims(f0,-1)
        vol = jnp.expand_dims(vol,-1)
        x = self.ppg_stack(ppg) + self.f0_embed(jnp.log(1+ f0 / 700)) + self.volume_embed(vol)
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
        x = self.decoder(x,train=train)
        x = self.norm(x)
        x = self.dense_out(x)
        return x

