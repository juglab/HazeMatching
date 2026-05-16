import types

from torch.utils.checkpoint import checkpoint
from torchcfm.models.unet.unet import UNetModel as _UNetBase

_NUM_CLASSES = 1000
 

def _enable_attention_checkpoint(model):
    """Patch all AttentionBlocks in model to always use gradient checkpointing."""
    for m in model.modules():
        if type(m).__name__ == "AttentionBlock":

            def _forward(self, x):
                return checkpoint(self._forward, x, use_reentrant=True)

            m.forward = types.MethodType(_forward, m)


class CCFMUNet(_UNetBase):
    """UNet wrapper used by CCFM.

    Differences from the upstream torchcfm UNetModelWrapper:
    - ``out_channels`` is an explicit parameter (upstream hardcodes it as ``dim[0]``).
    - All AttentionBlocks use gradient checkpointing unconditionally.

    Args:
        dim: ``(C, H, W)`` tuple describing the input tensor shape.
        num_channels: Base number of model channels.
        out_channels: Number of output channels.
        num_res_blocks: Number of residual blocks per resolution.
    """

    def __init__(
        self,
        dim,
        num_channels,
        out_channels,
        num_res_blocks,
        channel_mult=None,
        learn_sigma=False,
        class_cond=False,
        num_classes=_NUM_CLASSES,
        use_checkpoint=False,
        attention_resolutions="16",
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        dropout=0,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
    ):
        image_size = dim[-1]
        if channel_mult is None:
            if image_size == 512:
                channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
            elif image_size == 256:
                channel_mult = (1, 1, 2, 2, 4, 4)
            elif image_size == 128:
                channel_mult = (1, 1, 2, 3, 4)
            elif image_size == 64:
                channel_mult = (1, 2, 3, 4)
            elif image_size == 32:
                channel_mult = (1, 2, 2, 2)
            elif image_size == 28:
                channel_mult = (1, 2, 2)
            else:
                raise ValueError(f"unsupported image size: {image_size}")
        else:
            channel_mult = list(channel_mult)

        attention_ds = [image_size // int(r) for r in attention_resolutions.split(",")]

        super().__init__(
            image_size=image_size,
            in_channels=dim[0],
            model_channels=num_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            num_classes=(num_classes if class_cond else None),
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
        )
        _enable_attention_checkpoint(self)

    def forward(self, t, x, y=None, *args, **kwargs):
        return super().forward(t, x, y=y)
