import torch.nn.functional as F


def tpu_splash_attention(
    module,
    query,
    key,
    value,
    attention_mask,
    dropout=0.0,
    scaling=None,
    **kwargs,
):
    # Scoped to decider's recipe: causal, right-padded, unpacked training,
    # with use_cache=False and no attention dropout.
    if query.device.type != "tpu":
        raise RuntimeError("Splash Attention requires TPU tensors")
    if dropout != 0.0:
        raise ValueError("This Splash wrapper does not support dropout")
    if query.shape[-2] != key.shape[-2]:
        raise ValueError("This wrapper does not support cached decoding")
    if attention_mask is not None and attention_mask.ndim != 2:
        raise ValueError("Expected a 2D right-padding mask")

    # Keep the optional TorchTPU/JAX dependency out of CPU and SDPA imports.
    from torch_tpu._internal.ops.splash_attention import splash_sdpa

    length = query.shape[-2]

    # Conservative alignment for the default 512-token tiles.
    padding = (-length) % 512
    query = F.pad(query, (0, 0, 0, padding))
    key = F.pad(key, (0, 0, 0, padding))
    value = F.pad(value, (0, 0, 0, padding))

    output = splash_sdpa(
        query,
        key,
        value,
        scale=scaling,
        is_causal=True,
        enable_gqa=query.shape[1] != key.shape[1],
    )
    output = output[:, :, :length, :].transpose(1, 2)

    if attention_mask is not None:
        output = output * attention_mask[:, :, None, None].to(output.dtype)

    return output.contiguous(), None
