from functools import partial
import jax.numpy as jnp
import flax.linen as nn
import jax
class ConformerNaiveEncoder(nn.Module):
    """
    Conformer Naive Encoder

    Args:
        dim_model (int): Dimension of model
        num_layers (int): Number of layers
        num_heads (int): Number of heads
        expansion_factor (int): Expansion factor of conv module, default 2
        kernel_size (int): Kernel size of conv module, default 31
        use_norm (bool): Whether to use norm
        conv_only (bool): Whether to use only conv module without attention, default True
        conv_dropout (float): Dropout rate of conv module, default 0.
        atten_dropout (float): Dropout rate of attention module, default 0.
    """
    num_layers: int
    num_heads: int
    dim_model: int
    expansion_factor: int = 2
    kernel_size: int = 31
    use_norm: bool = False
    conv_only: bool = True
    conv_dropout: float = 0.
    atten_dropout: float = 0.1
    precision : jax.lax.Precision = jax.lax.Precision.HIGHEST

    @nn.compact
    def __call__(self, x, mask=None,train=True) -> jnp.ndarray:
        """
        Args:
            x (torch.Tensor): Input tensor (#batch, length, dim_model)
            mask (torch.Tensor): Mask tensor, default None
        return:
            torch.Tensor: Output tensor (#batch, length, dim_model)
        """
        x,_ = nn.scan(CFNEncoderLayer, 
                length=self.num_layers,
                variable_axes={"params": 0},
                split_rngs={"params": True},
                metadata_params={nn.PARTITION_NAME: None}
                )(dim_model=self.dim_model,
                expansion_factor=self.expansion_factor,
                kernel_size=self.kernel_size,
                num_heads=self.num_heads,
                conv_only=self.conv_only,
                conv_dropout=self.conv_dropout,
                atten_dropout=self.atten_dropout,
                precision=self.precision,
                train=train)(x)
        # for (i, layer) in enumerate(self.encoder_layers):
        #     x = layer(x, mask)
        return x  # (#batch, length, dim_model)


class CFNEncoderLayer(nn.Module):
    """
    Conformer Naive Encoder Layer

    Args:
        dim_model (int): Dimension of model
        expansion_factor (int): Expansion factor of conv module, default 2
        kernel_size (int): Kernel size of conv module, default 31
        num_heads (int): Number of heads
        use_norm (bool): Whether to use norm
        conv_only (bool): Whether to use only conv module without attention, default False
        conv_dropout (float): Dropout rate of conv module, default 0.1
        atten_dropout (float): Dropout rate of attention module, default 0.1
    """
    dim_model: int
    expansion_factor: int = 2
    kernel_size: int = 31
    num_heads: int = 8
    use_norm: bool = False
    conv_only: bool = True
    conv_dropout: float = 0.
    atten_dropout: float = 0.1
    precision : jax.lax.Precision = jax.lax.Precision.HIGHEST
    train:bool = True

    def setup(self):
        self.conformer = ConformerConvModule(
            self.dim_model,
            expansion_factor=self.expansion_factor,
            kernel_size=self.kernel_size,
            use_norm=self.use_norm,
            dropout=self.conv_dropout,
            precision=self.precision,
            train=self.train
        )

        self.norm = nn.LayerNorm()

    def __call__(self, x, mask=None):
        """
        Args:
            x (torch.Tensor): Input tensor (#batch, length, dim_model)
            mask (torch.Tensor): Mask tensor, default None
        return:
            torch.Tensor: Output tensor (#batch, length, dim_model)
        """
        x = self.norm(x)
        x = x + (self.conformer(x))

        return x,None 
class ConformerConvModule(nn.Module):
    dim:int
    expansion_factor:int=2
    kernel_size:int=31
    dropout:float=0.
    use_norm:bool=False
    precision : jax.lax.Precision = jax.lax.Precision.HIGHEST
    train:bool = True
    @nn.compact
    def __call__(self, x):
        inner_dim = self.dim * self.expansion_factor
        net = nn.Sequential([
            nn.Conv(inner_dim * 2, 1,precision=self.precision),
            partial(nn.glu,axis=2),
            nn.Conv(inner_dim, self.kernel_size, feature_group_count=inner_dim,precision=self.precision),
            nn.PReLU(),
            nn.Conv(self.dim, 1,precision=self.precision),
            nn.Dropout(self.dropout,deterministic=not self.train)]
        )
        return net(x)