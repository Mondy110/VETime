"""Smoke test for the VETime SRD preprocessing pipeline (dataset/pre_image.py).

Synthetic signal = slow trend + clean seasonal cycle + injected anomalies
(one single-point spike and one 20-point contextual burst). Verifies that the
new SRD pipeline (robust_iterative_decompose + RevIN + 3-sigma residual
enhancement) produces images of the right shape/dtype and yields an
anomaly-immune trend baseline, in contrast to the legacy moving average.
"""
import numpy as np
from dataset.pre_image import (
    ts2image_1d, ts2image_Test, moving_average_decompose,
    robust_iterative_decompose, _robust_mad_scale,
)


def make_signal(n=2000, m=50, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend = 0.0008 * t + 2.0 * np.sin(t / 800.0)            # smooth macro trend
    seasonal = 3.0 * np.sin(2 * np.pi * t / m)              # clean period m
    noise = rng.normal(0, 0.3, size=n)
    x = trend + seasonal + noise
    clean = x.copy()
    x[500] += 15.0                                          # single-point spike
    x[1400:1420] += 12.0                                    # contextual burst
    return x, clean, m


def main():
    x, clean, m = make_signal()

    # --- raw component checks ---
    mean, std = x.mean(), x.std() + 1e-8
    xn = (x - mean) / std
    cn = (clean - clean.mean()) / (clean.std() + 1e-8)

    t_srd, s_srd, r_srd = robust_iterative_decompose(xn, period=m)
    t_ma, r_ma = moving_average_decompose(xn.reshape(-1, 1), m)

    # Ground-truth trend = median-filtered CLEAN signal. The SRD trend on the
    # DIRTY signal should hug the clean trend far more tightly than MA near the
    # contextual burst, because the median is not bent by consecutive errors.
    from scipy.ndimage import median_filter
    true_trend = median_filter(cn, size=m, mode='nearest')
    win = slice(1390, 1430)
    ma_err = np.sqrt(np.mean((t_ma.ravel()[win] - true_trend[win]) ** 2))
    srd_err = np.sqrt(np.mean((t_srd[win] - true_trend[win]) ** 2))
    print(f"[trend robustness] RMS vs clean trend around burst: "
          f"MA={ma_err:.4f}, SRD={srd_err:.4f}")
    assert srd_err < ma_err, \
        "SRD trend should be closer to clean trend than MA near the burst"

    # SRD should push MORE anomaly mass into the residual than MA (less of the
    # burst absorbed into the trend), so the burst stays bright in the G channel.
    assert np.abs(r_srd[1400:1420]).mean() >= np.abs(r_ma.ravel()[1400:1420]).mean(), \
        "SRD residual should retain the burst (anomaly not absorbed into trend)"

    # --- _robust_mad_scale: outliers keep magnitude, in-band is dimmed ---
    scaled = _robust_mad_scale(r_srd, sigma=3.0, compress=0.1)
    assert abs(scaled[500]) >= abs(scaled[10]), \
        "Anomalous residual should remain brighter than normal residual"

    # --- ts2image_1d: shape / dtype / period contract ---
    # The padding-value contract is (C, 3) per-channel RGB means (matches the
    # original implementation): (1,3) univariate, (2,3) for 2 channels.
    for decomp in ('srd', 'ma'):
        for make_rgb in (True, False):
            img, period, pad = ts2image_1d(
                x, max_width=2016, patch_size=16, make_RGB=make_rgb,
                decomp_method=decomp,
            )
            assert img.shape == (3, 1, 2016), f"bad shape {img.shape} ({decomp}/{make_rgb})"
            assert img.dtype == np.uint8, f"bad dtype {img.dtype} ({decomp}/{make_rgb})"
            assert pad.shape == (1, 3), f"bad pad shape {pad.shape}"
            assert period >= 1
    print(f"[ts2image_1d] shape/dtype OK for srd+ma x rgb/grayscale")

    # --- multivariate (2 channels) ---
    x2 = np.stack([x, clean], axis=1)
    img, period, pad = ts2image_1d(x2, max_width=2016, patch_size=16)
    assert img.shape == (3, 2, 2016), f"bad multivariate shape {img.shape}"
    assert pad.shape == (2, 3)
    print(f"[ts2image_1d multivariate] shape {img.shape} OK")

    # --- ts2image_Test end-to-end ---
    img, period, pad = ts2image_Test(x, patch_size=16)
    assert img.shape[0] == 3 and img.shape[2] == 2016
    print(f"[ts2image_Test] shape {img.shape} OK")

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
