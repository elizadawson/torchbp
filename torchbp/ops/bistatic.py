"""Exact TX/target/RX geometry for direct BP and GPGA target histories."""

import math
import torch
from torch import Tensor
from ..grid import unpack_cartesian_grid, unpack_polar_grid


def _bistatic_look_vectors(targets, tx, rx):
    """Sensitivity of equivalent range to a shared platform displacement.

    Vectors point toward each target; the range derivative is their negative.
    Shape is [sweeps, targets, 3].
    """
    ut = targets[None, :, :] - tx[:, None, :]
    ur = targets[None, :, :] - rx[:, None, :]
    ut = ut / torch.linalg.vector_norm(ut, dim=-1, keepdim=True).clamp(min=1e-12)
    ur = ur / torch.linalg.vector_norm(ur, dim=-1, keepdim=True).clamp(min=1e-12)
    return (ut + ur) / 2


def _validate(targets, data, tx, rx, fc, r_res, d0, data_fmod):
    if data.ndim != 2 or data.shape[0] < 1 or data.shape[1] < 2:
        raise ValueError("bistatic data must have shape [sweeps, range bins], with at least two bins")
    if targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError("target_pos must have shape [points, 3]")
    if tx.shape != (data.shape[0], 3) or rx.shape != tx.shape:
        raise ValueError("TX and RX positions must both have shape [sweeps, 3]")
    if data.dtype not in (torch.complex64, torch.complex128):
        raise ValueError("bistatic data must be complex64 or complex128")
    for p in (targets, tx, rx):
        if p.device != data.device or p.dtype not in (torch.float32, torch.float64):
            raise ValueError("positions must be float32 or float64 on the data device")
        if not torch.isfinite(p).all():
            raise ValueError("positions must be finite")
    if not all(math.isfinite(v) for v in (fc, r_res, d0, data_fmod)) or fc <= 0 or r_res <= 0:
        raise ValueError("invalid frequency, range spacing, range origin, or modulation")


def _reference_samples(targets, data, tx, rx, fc, r_res, d0, data_fmod,
                       sum_sweeps, point_chunk=256, sweep_chunk=256):
    """Differentiable path, with geometry temporaries bounded per chunk.

    Forward calls without gradients use the compiled CPU/CUDA implementation.
    This path also supports float64 geometry for numerical verification.
    """
    blocks = []
    for p0 in range(0, len(targets), point_chunk):
        points = targets[p0:p0 + point_chunk]
        parts = []
        total = None
        for s0 in range(0, len(data), sweep_chunk):
            s1 = min(s0 + sweep_chunk, len(data))
            rt = torch.linalg.vector_norm(points[:, None] - tx[None, s0:s1], dim=-1)
            rr = torch.linalg.vector_norm(points[:, None] - rx[None, s0:s1], dim=-1)
            path = rt + rr
            u = (path / 2 + d0) / r_res
            valid = torch.isfinite(u) & (u >= 0) & (u < data.shape[1] - 1)
            safe_u = torch.where(valid, u, 0)
            lo = safe_u.long()
            f = safe_u - lo
            sweeps = torch.arange(s0, s1, device=data.device)[None, :]
            v = data[sweeps, lo] * (1 - f) + data[sweeps, lo + 1] * f
            phase = (2 * torch.pi * fc / 299792458.0) * path - data_fmod * safe_u
            v = torch.where(valid, v * torch.exp(1j * phase), 0)
            if sum_sweeps:
                contribution = v.sum(dim=1)
                total = contribution if total is None else total + contribution
            else:
                parts.append(v)
        blocks.append(total if sum_sweeps else torch.cat(parts, dim=1))
    if not blocks:
        shape = (0,) if sum_sweeps else (0, len(data))
        return data.new_empty(shape)
    return torch.cat(blocks, dim=0)


def _bistatic_samples(targets: Tensor, data: Tensor, tx: Tensor, rx: Tensor,
                      fc: float, r_res: float, d0: float = 0,
                      data_fmod: float = 0, sum_sweeps: bool = False) -> Tensor:
    _validate(targets, data, tx, rx, fc, r_res, d0, data_fmod)
    grad = torch.is_grad_enabled() and any(t.requires_grad for t in (targets, data, tx, rx))
    native = (data.dtype == torch.complex64 and
              all(t.dtype == torch.float32 for t in (targets, tx, rx)) and
              data.device.type in ("cpu", "cuda"))
    if native and not grad:
        return torch.ops.torchbp.bistatic_samples(
            targets, data, tx, rx, fc, r_res, d0, data_fmod, sum_sweeps)
    return _reference_samples(targets, data, tx, rx, fc, r_res, d0,
                              data_fmod, sum_sweeps)


def _grid_targets(grid, polar, device, dtype, dem=None):
    if polar:
        a0, _, b0, _, na, nb, da, db = unpack_polar_grid(grid)
    else:
        a0, _, b0, _, na, nb, da, db = unpack_cartesian_grid(grid)
    a = a0 + da * torch.arange(na, device=device, dtype=dtype)
    b = b0 + db * torch.arange(nb, device=device, dtype=dtype)
    aa, bb = torch.meshgrid(a, b, indexing="ij")
    x, y = (aa * torch.sqrt(torch.clamp(1 - bb**2, min=0)), aa * bb) if polar else (aa, bb)
    z = torch.zeros_like(x)
    if dem is not None:
        if dem.ndim != 2 or dem.device != device or not dem.is_floating_point():
            raise ValueError("dem must be a 2D floating-point tensor on the data device")
        # Match the existing BP kernel's pixel-index to DEM-index mapping.
        fi = torch.arange(na, device=device, dtype=dtype) * dem.shape[0] / na
        fj = torch.arange(nb, device=device, dtype=dtype) * dem.shape[1] / nb
        gi, gj = torch.meshgrid(fi, fj, indexing="ij")
        coords = torch.stack((2 * gj / max(1, dem.shape[1] - 1) - 1,
                              2 * gi / max(1, dem.shape[0] - 1) - 1), dim=-1)
        z = torch.nn.functional.grid_sample(
            dem.to(dtype)[None, None], coords[None], mode="bilinear",
            padding_mode="border", align_corners=True)[0, 0]
    return torch.stack((x, y, z), dim=-1).reshape(-1, 3), (na, nb)


def _backprojection_bistatic(data, grid, fc, r_res, tx_pos, rx_pos,
                             d0=0, data_fmod=0, polar=False, dem=None):
    points, shape = _grid_targets(grid, polar, data.device, tx_pos.dtype, dem)
    image = _bistatic_samples(points, data, tx_pos, rx_pos, fc, r_res,
                              d0, data_fmod, sum_sweeps=True)
    return image.reshape(1, *shape)


@torch.library.register_fake("torchbp::bistatic_samples")
def _fake_bistatic_samples(target_pos, data, tx_pos, rx_pos, fc, r_res,
                           d0, data_fmod, sum_sweeps):
    shape = (target_pos.shape[0],) if sum_sweeps else (target_pos.shape[0], data.shape[0])
    return data.new_empty(shape)
