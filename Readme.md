# Torchbp

![SAR image](https://github.com/Ttl/torchbp/blob/master/docs/img/07_19_1_autofocus_sigma0_pol_cal_pauli.png?raw=true)

Fast C++ PyTorch extension for differentiable synthetic aperture radar image formation and autofocus.

This is the personal fork [elizadawson/torchbp](https://github.com/elizadawson/torchbp)
of [Ttl/torchbp](https://github.com/Ttl/torchbp), used as a submodule by the
`radio-radar` SWW processing workflow. GPU acceleration requires an NVIDIA GPU;
CPU support depends on the operation.

Changes in this fork:

- Direct Cartesian and polar BP and GPGA target histories accept separate TX/RX
  positions through `rx_pos`, with compiled CPU/CUDA kernels and a PyTorch
  reference implementation.
- GPGA-TDE estimates a common platform translation for both antennas, preserving
  their baseline. The underlying GPGA-TDE algorithm comes from upstream.
- GPGA and GPGA-TDE accept actual `sweep_times` for filtering and detrending
  uneven acquisitions. Basic GPGA rejects separate RX positions; use GPGA-TDE
  for bistatic trajectory correction.
- GPGA-TDE optionally fills `rms_history` with each iteration's phase-equivalent
  RMS position correction in radians. This measures convergence, not true
  residual phase error.
- Editable installation supports imports from a parent repository containing
  the `torchbp` submodule without a `PYTHONPATH` override.

The bistatic path currently supports direct BP with linear interpolation.
Factorized BP and minimum-entropy wrappers have not been extended for separate
TX/RX geometry. Calls without `rx_pos` retain the upstream monostatic path.
See [bistatic processing](docs/bistatic.md) for conventions and supported options.
The CPU physics checks have passed on macOS; the added CUDA kernel has not
been built or tested on NVIDIA hardware in this workflow.

On RTX 3090 Ti backprojection on polar grid achieves 370 billion
backprojections/s and fast factorized backprojection (ffbp) is ten times faster on
moderate size images.

## Installation

Tested with CUDA version 12.9.

### From source

Install PyTorch first, then build torchbp against that same torch with
build isolation disabled:

```bash
pip install torch
git clone https://github.com/elizadawson/torchbp.git
cd torchbp
pip install --no-build-isolation -e .
```

`--no-build-isolation` is required. The extension links libtorch C++ symbols
that are not ABI-stable across torch versions, so it must be compiled against
the exact torch you run at import time. Without the flag, pip builds in an
isolated environment with a different (freshly downloaded) torch, producing
a `.so` that fails to import with an `undefined symbol` error. For the same
reason, rebuild torchbp whenever you change your torch version.

## Tests

Install test dependencies, then run pytest:

```bash
pip install --no-build-isolation -e .[test]
pytest tests/
```

On a CPU-only machine the CUDA tests are skipped rather than failing.

## Documentation

Upstream documentation can be viewed at: https://ttl.github.io/torchbp/
Fork-specific bistatic behavior is documented in [docs/bistatic.md](docs/bistatic.md).

API documentation and examples can be built with sphinx.

```bash
pip install --no-build-isolation .[docs]
cd docs
make html
```

Open `docs/build/html/index.html`.
