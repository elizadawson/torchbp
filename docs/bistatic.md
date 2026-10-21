# Bistatic backprojection and autofocus

Direct Cartesian and polar backprojection, GPGA target extraction, and
`gpga_tde` accept an optional `rx_pos`. With it, the existing `pos` argument is
the transmitter position. Both position arrays have shape `[sweeps, 3]` in the
same local Cartesian frame. Input data is a complex array `[sweeps, range bins]`.

The implementation uses the full propagation path:

```
L = norm(target - tx_pos) + norm(target - rx_pos)
sample_index = (L/2 + d0) / r_res
carrier = exp(+j * 2*pi*fc*L/c)
```

`r_res` is range-bin spacing, not necessarily physical range resolution. Bins
represent equivalent one-way range `c*delay/2`. For calibrated input whose
first bin is at `r_first`, set `d0=-r_first`. `fc` must match the phase reference
of the supplied complex profiles. `data_fmod` retains its existing meaning.
Already range-compressed profiles do not need the FMCW preprocessing used in
the sample-data examples.

```python
image = torchbp.ops.backprojection_cart_2d(
    profiles, grid, fc, range_spacing, tx_pos,
    rx_pos=rx_pos, d0=-first_range,
)

image, tx_refined = torchbp.autofocus.gpga_tde(
    None, profiles, tx_pos, fc, range_spacing, grid,
    azimuth_divisions=4, range_divisions=4,
    rx_pos=rx_pos, sweep_times=relative_sweep_times,
    d0=-first_range, algorithm="bp",
)
rx_refined = rx_pos + (tx_refined - tx_pos)
```

Nominal BP supports distinct, arbitrary TX and RX trajectories. Autofocus
estimates a **common platform translation** per sweep and applies it to both
antennas, preserving their known per-sweep baseline. This models antennas
mounted on the same platform. `gpga_tde` uses the mean of the two antenna look
vectors in its equivalent-range position solve. Basic `gpga` retains a
monostatic phase-to-position approximation and rejects `rx_pos`; use
`gpga_tde` for bistatic position corrections.

Supply actual `sweep_times` for uneven timing. Times must be strictly increasing;
use relative seconds or float64 absolute timestamps. Only geometry-compensated
target histories are interpolated to a regular time grid for low-pass filtering,
then evaluated at the measured sweep times. Detrending also uses those times.
The measured profiles and positions are never regridded. Without this argument,
the previous sample-index filtering and detrending are retained. Large gaps can
still make phase differences ambiguous; timestamps do not recover missing echoes.

The first bistatic implementation supports direct BP with linear interpolation,
one acquisition at a time, and optional polar DEM heights. No-gradient float32
geometry/complex64 data use compiled CPU/CUDA kernels. Differentiable calls and
float64 inputs use a chunked PyTorch implementation. Native BP accumulates
directly into the image and does not allocate a pixel-by-sweep geometry cube.

Fast factorized BP, antenna-pattern weighting, beam gating, image dealiasing,
the coarse-aspect projection, and the minimum-entropy wrappers have not yet
been extended for separate antenna geometry. Unsupported options on the
bistatic BP/GPGA path raise errors instead of falling back to monostatic math.
Omitting `rx_pos` continues to use the existing monostatic implementation.

Build the extension again after updating the submodule sources. Run the physics
checks with `python -m pytest tests/test_bistatic.py` from the submodule root.
They cover a curved irregular aperture, nonzero range origin, skipped sweeps,
TX/RX symmetry, the monostatic limit, position-solve sensitivity, and derivatives
with respect to both antennas. CUDA checks are skipped when no CUDA device exists.
