import torch


def add_device_arg(parser):
    """Adds a shared -d/--device flag to an argparse parser."""
    parser.add_argument(
        "-d",
        "--device",
        choices=["gpu", "cpu"],
        default="gpu",
        help="Device to run inference on (default: gpu).",
    )


def resolve_device(device: str) -> str:
    """Resolves a --device flag value ("gpu"/"cpu") to a torch/safetensors device string."""
    if device == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU requested but CUDA is not available. Pass --device cpu / -d cpu to run on CPU."
            )
        return "cuda"
    elif device == "cpu":
        return "cpu"
    else:
        raise ValueError(f"Unknown device '{device}', expected 'gpu' or 'cpu'.")


def bytes_to_unicode(reverse=False):
    """
    Reference implementation for mapping bytes to unicode values
    from gpt2.
    https://github.com/openai/gpt-2/blob/master/src/encoder.py
    """
    # These ranges are unicode-nice already and don't need to be mapped.
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs.copy()
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            # non-nice characters are simply shifted up by 256.
            # The reasonable interpretation is that this prevents collision
            # and all unicode 256-512 are cleanly printable.
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    if reverse:
        return dict(zip(cs, bs))
    else:
        return dict(zip(bs, cs))


bytes_to_unicode_map = bytes_to_unicode()
reverse_bytes_to_unicode_map = bytes_to_unicode(reverse=True)
