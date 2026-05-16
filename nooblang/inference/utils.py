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
