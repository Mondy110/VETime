"""
Time series to image conversion utilities for VETime.

This module provides functions to convert 1D time series data into 2D image
representations for vision-based anomaly detection. The conversion leverages
periodicity detection and trend-residual decomposition to create meaningful
visual representations.

Example:
    >>> from dataset.pre_image import ts2image_Test
    >>> ts = np.random.randn(1000, 1)
    >>> img, period, pad_value = ts2image_Test(ts, patch_size=16, img_size=224)
"""
import numpy as np
import torch
import math
from typing import Tuple, Union, List, Optional
from statsmodels.tsa.stattools import acf
from scipy.signal import argrelextrema
from scipy.ndimage import median_filter


def ts2image_Test(
    x: Union[np.ndarray, torch.Tensor],
    patch_size: int,
    T_sqrt: bool = False,
    img_size: int = 224,
    make_RGB: bool = True,
    decomp_method: str = 'srd',
    max_iter: int = 3,
    robust_sigma: float = 3.0,
    compress_ratio: float = 0.1
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Convert 1D time series to image representation for testing/evaluation.

    This function transforms time series data into a 2D image by:
    1. Detecting periodicity in each channel using autocorrelation
    2. Computing a global period across all channels
    3. Converting to image using ts2image_1d、

    Args:
        x: Input time series data. Shape (L, C) for multivariate or (L,) for 
           univariate time series, where L is sequence length and C is number 
           of channels.
        patch_size: Size of each patch for dividing the time series. Used in 
                    period calculation and image construction.、
        T_sqrt: If True, use sqrt(T) for period height calculation where T is 
                the number of patches. If False, use detected global period.
                Default: False.
        make_RGB: If True, create RGB image using seasonal-trend decomposition
                  (original + trend + residual channels). If False, create 
                  grayscale-like 3-channel image. Default: True.
        decomp_method: Decomposition backend ('srd' robust iterative SRD, or
                       'ma' legacy moving average). Default: 'srd'.
        max_iter: Maximum SRD repair iterations. Default: 3.
        robust_sigma: 3-sigma threshold multiplier. Default: 3.0.
        compress_ratio: Darkening factor for in-threshold residual pixels.
                        Default: 0.1.

    Returns:
        img: Image array of shape (3, L, img_size), dtype float32.
             Contains the time series visualized as an image.
        period: Detected global period (integer), representing the dominant 
                periodicity in the time series.
        pad_value: Padding values of shape (3, C, 1) used for extending the 
                   time series to fit the image dimensions.
    """
    if x.ndim == 1:
        x = x[:, np.newaxis]
    L, C = x.shape
    
    periods_per_channel = []
    for c in range(C):
        xc = x[:, c].copy()
        mean = xc.mean()
        std = xc.std() + 1e-8
        xc_norm = (xc - mean) / std
        period = find_period(xc_norm)
        periods_per_channel.append(period)
    
    global_period = int(np.max(periods_per_channel)) if periods_per_channel else 1
    global_period = max(1, global_period)
    
    lengths = x.shape[0]
    max_width = ((lengths + patch_size-1) // patch_size) * patch_size

    img, period, pad_values = ts2image_1d(
        x, max_width, patch_size, h_size=1, make_RGB=make_RGB,
        decomp_method=decomp_method, max_iter=max_iter,
        robust_sigma=robust_sigma, compress_ratio=compress_ratio,
    )

    
    return img, period, pad_values


def ts2image_1d(
    x: Union[np.ndarray, torch.Tensor],
    max_width: int,
    patch_size: int,
    h_size: int = 1,
    make_RGB: bool = True,
    decomp_method: str = 'srd',
    max_iter: int = 3,
    robust_sigma: float = 3.0,
    compress_ratio: float = 0.1
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Convert time series to RGB image representation with trend/residual decomposition.

    This is the core function that transforms 1D time series into a 2D image by:
    1. Normalizing each channel using z-score normalization
    2. Detecting global periodicity across all channels
    3. Applying moving average decomposition (for RGB mode)
    4. Tiling and stacking channels vertically
    5. Applying gamma correction for visual enhancement

    The output image encodes the time series as a heatmap where:
    - For RGB mode: R=original, G=residual (3-sigma enhanced), B=trend components
    - For non-RGB mode: All three channels contain the normalized original signal

    Args:
        x: Input time series data. Shape (L, C) for multivariate or (L,) for 
           univariate time series, where L is sequence length and C is number 
           of channels.
        max_width: Target width for the output image. The time series will be 
                   padded or truncated to this length.
        patch_size: Size of each patch for dividing the time series. Used in 
                    period detection and image construction.
        h_size: Height multiplier for each channel. Each channel will be 
                repeated h_size times vertically. Default: 1.
        make_RGB: If True, create RGB image using seasonal-trend decomposition
                  with three components (original, residual, trend). If False,
                  replicate the normalized signal across all three channels.
                  Default: True.
        decomp_method: Decomposition backend for the RGB channels.
            - 'srd': Robust iterative SRD decomposition (Algorithm 1 of the
              SIGMOD'25 paper). Anomaly-immune median trend + same-phase
              seasonal + boundary extension + 3-sigma repair. Produces a
              straight, uncontaminated trend baseline. Default.
            - 'ma': Legacy moving-average trend/residual decomposition.
        max_iter: Maximum number of SRD repair iterations (only used when
                  decomp_method='srd'). Default: 3.
        robust_sigma: Multiplier for the 3-sigma threshold used both in SRD
                      violation detection and in the non-linear residual
                      contrast enhancement. Default: 3.0.
        compress_ratio: Darkening factor applied to in-threshold residual
                        pixels in the G channel (outliers stay bright).
                        Default: 0.1.

    Returns:
        final_image: Image array of shape (3, C * h_size, max_width), dtype 
                     uint8. Each channel of the time series is converted to a 
                     3-channel image tile and stacked vertically.
        period: Detected global period (integer), representing the maximum 
                periodicity across all channels.
        pad_values: Padding values of shape (3, C, 1), dtype uint8. Contains 
                    the mean values used for padding each channel, scaled to 
                    [0, 255].
    """
    if x.ndim == 1:
        x = x[:, np.newaxis]
    L, C = x.shape
    
    gamma_L = np.linspace(0.5, 1.5, C).tolist()
    np.random.shuffle(gamma_L)
    
    periods_per_channel = []
    for c in range(C):
        xc = x[:, c].copy()
        mean = xc.mean()
        std = xc.std() + 1e-8
        xc_norm = (xc - mean) / std
        period = find_period(xc_norm)
        periods_per_channel.append(period)
    
    global_period = int(np.max(periods_per_channel)) if periods_per_channel else 1
    global_period = max(1, global_period)
    
    pad_value_l = []
    images_per_channel = []
    
    for c in range(C):
        xc = x[:, c].copy()

        # Step 1: Homogenization — RevIN (reversible instance normalization).
        # Per-series affine z-score: removes cross-dataset scale/variance
        # heterogeneity so the vision model learns a single anomaly signature.
        # Only the forward pass is needed (the image is never inverted back).
        mean = xc.mean(axis=0, keepdims=True)
        std = xc.std(axis=0, keepdims=True) + 1e-8
        xc_norm = ((xc - mean) / std).reshape(-1, 1)

        period = global_period

        if make_RGB:
            if decomp_method == 'srd':
                # Steps 2 & 3: Robust iterative SRD decomposition.
                # Produces an anomaly-immune (T_pure, S_pure) baseline and the
                # clean residual R = X - T_pure - S_pure.
                t_pure, s_pure, residual = robust_iterative_decompose(
                    xc_norm.ravel(),
                    period=period,
                    max_iter=max_iter,
                    sigma=robust_sigma,
                )
                chan_trend = t_pure
                chan_resid = residual
                # Step 4 (G channel): non-linear 3-sigma contrast enhancement.
                # Normal fluctuations dimmed by compress_ratio; anomalies keep
                # their magnitude (a spatial attention mask for the encoder).
                chan_resid = _robust_mad_scale(
                    chan_resid, sigma=robust_sigma, compress=compress_ratio
                )
            else:
                # Legacy moving-average baseline (trend, residual).
                chan_trend, chan_resid = moving_average_decompose(xc_norm, period)

            # Step 4: RGB mapping — R=original (visual anchor), G=enhanced
            # residual (attention guidance), B=pure trend (robust baseline).
            R = (xc_norm - xc_norm.min()) / (xc_norm.max() - xc_norm.min() + 1e-5)
            G = (chan_resid - chan_resid.min()) / (chan_resid.max() - chan_resid.min() + 1e-5)
            B = (chan_trend - chan_trend.min()) / (chan_trend.max() - chan_trend.min() + 1e-5)
            img_rgb = np.stack([R[..., 0], G.ravel(), B.ravel()], axis=-1)
        else:
            xc_vis = (xc_norm - xc_norm.min()) / (xc_norm.max() - xc_norm.min() + 1e-5)
            img_rgb = np.repeat(xc_vis[:, np.newaxis], 3, axis=-1)
        
        if img_rgb.shape[0] < max_width:
            img_rgb, pad_value = adaptive_pad_heatmap(img_rgb, max_width=max_width, period=period)
        else:
            img_rgb = img_rgb[:max_width]
            tail_window = max(5, period)
            pad_value = np.mean(img_rgb[-tail_window:], axis=0)
        
        img_tile = np.repeat(img_rgb[np.newaxis, :, :], h_size, axis=0)
        img_tile = np.transpose(img_tile, (2, 0, 1))
        img_tile = np.power(img_tile, gamma_L[c])
        images_per_channel.append(img_tile)
        pad_value_l.append(pad_value)
    
    final_image = np.concatenate(images_per_channel, axis=1)
    final_image = (final_image * 255).clip(0, 255).astype(np.uint8)
    pad_values = (np.stack(pad_value_l) * 255).clip(0, 255).astype(np.uint8)
    
    return final_image, period, pad_values


def adaptive_pad_heatmap(
    img_rgb: np.ndarray,
    max_width: int,
    period: int = 0,
    noise_ratio: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adaptively pad heatmap to target width while preserving periodic patterns.

    This function extends a time series heatmap to the target width by:
    1. If period is detected: Repeating the last periodic segment to maintain
       the natural pattern of the time series
    2. If no period: Using tail window averaging for constant padding

    This approach avoids introducing artificial patterns that could be 
    misinterpreted as anomalies during model inference.

    Args:
        img_rgb: Input heatmap array. Can be:
            - 2D array of shape (H, W) for single-channel
            - 3D array of shape (H, W, C) for multi-channel RGB
        max_width: Target height after padding. If current height H is already
                   >= max_width, no padding is applied.
        period: Period length detected from the time series. If period > 5,
                periodic padding is used; otherwise, constant padding is applied.
                Default: 0.
        noise_ratio: Ratio of noise to add for simulating normal fluctuations.
                     Currently reserved for future use. Default: 0.05.

    Returns:
        padded_img: Padded heatmap array of shape (max_width, W) for 2D input 
                    or (max_width, W, C) for 3D input. Same dtype as input.
        padding_value: Mean value of the padding content. Shape (W,) for 2D 
                       or (W, C) for 3D, representing the average value added 
                       during padding.
    """
    H = img_rgb.shape[0]
    pad_len = max_width - H
    
    if period > 5 and pad_len > 0:
        start_idx = max(0, H - period - pad_len)
        template = img_rgb[start_idx:H - pad_len]
        repeats = int(np.ceil(pad_len / period))
        extended = np.tile(template, (repeats, 1)) if img_rgb.ndim == 2 else np.tile(template, (repeats, 1, 1))
        padding_content = extended[:pad_len]
    else:
        tail_window = max(5, period)
        tail_mean = np.mean(img_rgb[-tail_window:], axis=0, keepdims=True)
        padding_content = np.tile(tail_mean, (pad_len, 1)) if img_rgb.ndim == 2 else np.tile(tail_mean, (pad_len, 1, 1))
    
    padded_img = np.concatenate([img_rgb, padding_content], axis=0)
    padding_value = np.mean(padding_content, axis=0)
    
    return padded_img.astype(img_rgb.dtype), padding_value


def find_period(
    data: Union[np.ndarray, torch.Tensor],
    top_k: int = 1,
    max_lag_ratio: float = 0.2
) -> int:
    """
    Estimate the period of time series data using autocorrelation function (ACF).

    This function detects periodicity by:
    1. Computing the autocorrelation function (ACF) of the centered time series
    2. Finding local maxima in the ACF values (excluding lag 0)
    3. Selecting the top-k peaks by ACF value strength
    4. Returning the lag corresponding to the k-th strongest peak

    The ACF is computed efficiently using FFT for long sequences.

    Args:
        data: Input time series data. Shape (L,) or (L, 1), where L is the 
              sequence length. If torch.Tensor, it will be converted to numpy.
        top_k: Which local maximum peak to select as the period. 
               top_k=1 returns the strongest peak, top_k=2 the second strongest,
               etc. Higher values may capture sub-harmonics. Default: 1.
        max_lag_ratio: Maximum lag as a ratio of sequence length for ACF 
                       calculation. Controls the maximum detectable period.
                       Default: 0.2 (i.e., max period = 20% of sequence length).

    Returns:
        estimated_period: Detected period as an integer. Returns 1 if:
            - No local maxima found in ACF
            - Strongest ACF peak < 0.5 (weak periodicity)
            - Sequence too short for reliable detection
    """
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    data = data.squeeze()
    
    data = data[:min(20000, len(data))]
    max_lag = int(min(20000, len(data)) * max_lag_ratio)
    if max_lag < 1:
        return 1
    
    auto_corr = acf(data - data.mean(), nlags=max_lag, fft=True)
    acf_vals = auto_corr[1:]
    lags = np.arange(1, max_lag + 1)
    local_max_indices = argrelextrema(acf_vals, np.greater)[0]
    
    if len(local_max_indices) == 0:
        return 1
    
    candidate_lags = lags[local_max_indices]
    candidate_values = acf_vals[local_max_indices]
    sorted_idx = np.argsort(candidate_values)[::-1]
    candidate_lags = candidate_lags[sorted_idx]
    valid_lags = candidate_lags[(candidate_lags >= 1) & (candidate_lags <= max_lag)]
    candidate_values = candidate_values[sorted_idx]
    
    if len(valid_lags) == 0 or candidate_values[0] < 0.5:
        return 1
    
    top_k = min(top_k, len(valid_lags))
    estimated_period = int(valid_lags[top_k - 1])
    
    return estimated_period


def moving_average_decompose(
    X: np.ndarray,
    K: int = 25
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose time series into trend and residual components using moving average.

    This function performs classical time series decomposition:
    1. Apply symmetric padding to handle boundary effects
    2. Compute moving average with a sliding window of size K
    3. Extract trend as the smoothed component
    4. Compute residual as the difference between original and trend

    The decomposition follows: X = trend + residual

    Args:
        X: Input time series data. Shape (T, D) for multivariate or (T,) for 
           univariate time series, where T is sequence length and D is number 
           of channels/features.
        K: Window size for moving average. Larger values produce smoother trends
           but may oversmooth short-term patterns. If K=1, defaults to window 
           size 25. Window size is adjusted to be odd if even. Default: 25.

    Returns:
        trend: Trend component of same shape as input X. Represents the 
               long-term smoothed pattern extracted via moving average.
        residual: Residual component of same shape as input X. Represents 
                  the short-term fluctuations and noise after removing trend.
                  Computed as: residual = X - trend
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    T, D = X.shape
    
    if K == 1:
        kernel_size = 25
    else:
        kernel_size = K
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
    pad_len = (kernel_size - 1) // 2
    
    trend = np.zeros_like(X)
    
    if pad_len > 0:
        X_padded = np.pad(X, ((pad_len, pad_len), (0, 0)), mode='reflect')
    else:
        X_padded = X.copy()
    
    for i in range(D):
        col = X_padded[:, i]
        window = np.ones(kernel_size) / kernel_size
        conv_result = np.convolve(col, window, mode='valid')
        trend[:, i] = conv_result
    
    residual = X - trend
    return trend, residual


def _robust_mad_scale(
    r: np.ndarray,
    sigma: float = 3.0,
    compress: float = 0.1
) -> np.ndarray:
    """
    Non-linear contrast enhancement of residuals via robust 3-sigma (MAD).

    This implements the "attention guidance" idea of Step 4: normal
    fluctuations (within `sigma` robust standard deviations) are darkened by
    a factor of `compress`, while suspected anomalies (beyond the threshold)
    keep their magnitude. The result is then clipped to [-1, 1] and the sign
    preserved, so a downstream min-max to [0, 1] turns anomalies into bright
    pixels and normal regions into dim ones.

    The robust scale is the standard MAD-based estimator:
        MAD = median(|r - median(r)|)
        sigma_hat = 1.4826 * MAD
        threshold T = sigma * sigma_hat

    Args:
        r: 1D residual array (already de-trended and de-seasonalized).
        sigma: Multiplier for the robust threshold (the "k" in k-sigma).
               Default: 3.0.
        compress: Darkening factor applied to in-threshold residuals.
                  Default: 0.1.

    Returns:
        Enhanced 1D array of the same shape, with anomalies preserved and
        normal regions attenuated, clipped to [-1, 1].
    """
    r = np.asarray(r, dtype=np.float64).ravel()
    med = np.median(r)
    mad = np.median(np.abs(r - med))
    sigma_hat = 1.4826 * mad if mad > 1e-8 else (r.std() + 1e-8)
    threshold = sigma * sigma_hat

    enhanced = r.copy()
    in_band = np.abs(r) <= threshold
    if compress != 1.0:
        enhanced[in_band] = enhanced[in_band] * compress
    # Anomalies keep their value but are clipped to [-1, 1] for stable imaging.
    enhanced = np.clip(enhanced, -1.0, 1.0)
    return enhanced


def robust_iterative_decompose(
    x: np.ndarray,
    period: int,
    max_iter: int = 3,
    sigma: float = 3.0,
    random_state: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Robust iterative seasonal-trend decomposition (SRD, Algorithm 1).

    Implements the SIGMOD'25 paper "Cleaning Time Series under Seasonal and
    Trend Constraints": a violation-tolerant median trend filter (Formula 4),
    a same-phase median seasonality filter (Formula 5), trend component
    extension at the boundaries (Formulas 6/7), and iterative repair of
    detected violations (Formula 8, 3-sigma detection). The result is an
    anomaly-immune (T_pure, S_pure) baseline and the final residual.

    Pipeline (each iteration h):
        1. Decomposition: median trend t, de-trend d = x - t, same-phase
           median seasonal s, boundary trend extension.
        2. Detection: residual r = x - t - s; eta = sigma * std(r);
           violations V = {|r| > eta}.
        3. Repair: if V non-empty, x'[i] = t[i] + s[i] + eps for i in V,
           eps ~ N(0, (eta/3)^2); loop with x'. Else converge.

    Args:
        x: 1D time series (already normalized, e.g. via RevIN z-score).
        period: Seasonal period m (>= 3). Values < 3 fall back to a
                fixed-window median trend with no seasonality.
        max_iter: Maximum number of repair iterations. The paper reports
                  convergence in ~4 rounds at 5 per-mille error rate.
                  Default: 3.
        sigma: Multiplier for the 3-sigma violation threshold. Default: 3.0.
        random_state: Seed for the repair white noise (set None for
                      non-determinism). Default: 42.

    Returns:
        A tuple (t_pure, s_pure, residual) of 1D float64 arrays, each of the
        same length as the input x:
            - t_pure: anomaly-immune pure trend component.
            - s_pure: anomaly-immune pure seasonal component.
            - residual: x - t_pure - s_pure over the (last-iterated) input.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.shape[0]
    rng = np.random.default_rng(random_state)

    # Degenerate cases: no usable periodicity or too-short series.
    if n < 3 or period < 3:
        win = max(3, min(25, n if n % 2 == 1 else n - 1))
        win = win + 1 if win % 2 == 0 else win
        t = median_filter(x, size=win, mode='nearest')
        s = np.zeros_like(x)
        return t, s, x - t - s

    m = int(period)
    half = (m - 1) // 2  # floor((m-1)/2), window half-width per Formula 4

    x_cur = x.copy()
    t = s = None

    def _seasonal(detrended: np.ndarray) -> np.ndarray:
        """Same-phase median seasonality (Formula 5), tiled to full length."""
        s_template = np.empty(m, dtype=np.float64)
        # phase p collects all d[j] with j ≡ p (mod m), p in [0, m)
        for p in range(m):
            vals = detrended[p::m]
            s_template[p] = np.median(vals) if vals.size else 0.0
        reps = int(np.ceil(n / m))
        s_full = np.tile(s_template, reps)[:n]
        return s_full

    def _trend_with_extension(series: np.ndarray) -> np.ndarray:
        """Median trend (Formula 4) + boundary extension (Formulas 6/7).

        The sliding median is incomplete in the first/last `half` points
        because there are not enough neighbours. We synthesize those boundary
        points from neighbouring-period trend + current seasonal phase
        (following the paper's component extension), then re-run the median
        so the trend is defined (and physically natural) across the whole span.
        """
        t_inner = median_filter(series, size=m, mode='nearest')
        s_local = _seasonal(series - t_inner)

        # Build an extended series: prepend/append half a period of synthetic
        # boundary points derived from the extended trend + seasonal phase,
        # then take a median filter over the extended span and crop back.
        # Leading points (i in 1..half): anchored to t_inner[half] + s_local
        if half > 0:
            lead_anchor = t_inner[half] if half < n else t_inner[-1]
            lead_phase = s_local[m - half:m] if m - half >= 0 else s_local[:half]
            lead = lead_anchor + lead_phase
            trail_anchor = t_inner[n - 1 - half] if n - 1 - half >= 0 else t_inner[0]
            trail_phase = s_local[:half]
            trail = trail_anchor + trail_phase
            extended = np.concatenate([lead, series, trail])
            t_ext = median_filter(extended, size=m, mode='nearest')
            t_full = t_ext[half:half + n]
        else:
            t_full = t_inner
        return t_full

    for _ in range(max_iter):
        t = _trend_with_extension(x_cur)
        s = _seasonal(x_cur - t)
        residual = x_cur - t - s

        std_r = residual.std()
        eta = sigma * std_r if std_r > 1e-8 else 0.0
        if eta == 0.0:
            break
        violations = np.abs(residual) > eta
        if not violations.any():
            break
        # Seasonal repair (Formula 8): replace violators by t + s + white noise.
        noise = rng.normal(0.0, eta / 3.0, size=int(violations.sum()))
        x_cur = x_cur.copy()
        x_cur[violations] = t[violations] + s[violations] + noise

    # Final decomposition on the last-iterated (possibly repaired) series, so
    # the returned T/S baseline is computed over the cleanest available input.
    t_pure = _trend_with_extension(x_cur)
    s_pure = _seasonal(x_cur - t_pure)
    residual = x - t_pure - s_pure
    return t_pure, s_pure, residual
