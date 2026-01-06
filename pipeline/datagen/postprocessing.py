import numpy as np

def crop_ok_lorenz63_two_lobes(
    crop: np.ndarray,
    *,
    axis: int=0,
    min_switches: int=2,
    min_lobe_fraction: float=0.15,
) -> bool:
    x = crop[:, axis]
    pos = x > 0
    pos_ct = int(pos.sum())
    neg_ct = crop.shape[0] - pos_ct
    if pos_ct == 0 or neg_ct == 0:
        return False

    switches = int((pos[1:] != pos[:-1]).sum())
    if switches < min_switches:
        return False

    min_ct = int(np.ceil(min_lobe_fraction * crop.shape[0]))
    if pos_ct < min_ct or neg_ct < min_ct:
        return False

    return True