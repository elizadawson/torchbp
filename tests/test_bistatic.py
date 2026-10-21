"""Physics checks for exact bistatic BP and the GPGA position model."""

import unittest
import math
import torch
import torchbp
from torchbp.ops.bistatic import _bistatic_look_vectors
from torchbp.autofocus import _sweep_coordinates, _detrend_history

C0 = 299792458.0


def point_scene(device="cpu"):
    # Uneven timing, a curved track, changing height, and a varying baseline.
    t = torch.linspace(0, 1, 29, dtype=torch.float64, device=device)**1.4
    tx = torch.stack((0.6 * torch.sin(3*t), 6*t-3, 8+0.3*torch.cos(2*t)), dim=1).float()
    baseline = torch.stack((0.5+0.2*t, 4+0.3*t, 1.5+0.1*t), dim=1).float()
    rx = tx + baseline
    target = torch.tensor([[40.0, 4.0, 0.0]], device=device)
    # Generate profiles from the physical TX-target-RX propagation path.
    path = (torch.linalg.vector_norm(target.double()-tx.double(), dim=1) +
            torch.linalg.vector_norm(target.double()-rx.double(), dim=1))
    fc, dr, first_range = 5.6e9, 0.04, 20.0
    r = first_range + dr*torch.arange(800, device=device, dtype=torch.float64)
    envelope = torch.exp(-0.5*((r[None, :]-path[:, None]/2)/0.20)**2)
    data = (envelope*torch.exp(-2j*torch.pi*fc*path[:, None]/C0)).to(torch.complex64)
    return t, tx, rx, target, data, fc, dr, -first_range


class BistaticChecks:
    device = "cpu"

    def test_point_target_focuses_with_exact_path(self):
        _, tx, rx, target, data, fc, dr, d0 = point_scene(self.device)
        history = torchbp.ops.gpga_backprojection_2d_core(
            target, data, tx, fc, dr, d0=d0, rx_pos=rx)
        self.assertGreater(history.abs().min().item(), 0.99)
        self.assertLess(history.angle().abs().max().item(), 2e-5)
        grid = {"x": (40.0, 40.5), "y": (4.0, 4.5), "nx": 1, "ny": 1}
        img = torchbp.ops.backprojection_cart_2d(data, grid, fc, dr, tx, d0=d0, rx_pos=rx)
        torch.testing.assert_close(img[0, 0, 0], history.sum(), rtol=2e-6, atol=2e-6)
        radius = math.hypot(40, 4)
        polar = {"r": (radius, radius+0.5), "theta": (4/radius, 4/radius+0.01),
                 "nr": 1, "ntheta": 1}
        img_polar = torchbp.ops.backprojection_polar_2d(
            data, polar, fc, dr, tx, d0=d0, rx_pos=rx)
        self.assertGreater(img_polar.abs().item(), 0.99*len(data))
        self.assertLess(img_polar.angle().abs().item(), 5e-3)

    def test_carrier_and_range_modulation_at_known_delay(self):
        tx = torch.tensor([[0., 0., 0.]], device=self.device)
        rx = torch.tensor([[10., 0., 0.]], device=self.device)
        point = torch.tensor([[30., 0., 0.]], device=self.device)
        fc, fmod = 1e8, 0.4
        # Total path 50 m, equivalent range 25 m, cropped first bin 20 m.
        j = torch.arange(16, dtype=torch.float64, device=self.device)
        data = torch.exp(-2j*torch.pi*fc*50/C0 + 1j*fmod*j)[None].to(torch.complex64)
        value = torchbp.ops.gpga_backprojection_2d_core(
            point, data, tx, fc, 1., d0=-20., data_fmod=fmod, rx_pos=rx)
        torch.testing.assert_close(value, torch.ones_like(value), rtol=1e-6, atol=1e-6)

    def test_tx_rx_exchange_and_skipped_sweeps(self):
        _, tx, rx, target, data, fc, dr, d0 = point_scene(self.device)
        f = torchbp.ops.gpga_backprojection_2d_core
        expected = f(target, data, tx, fc, dr, d0=d0, rx_pos=rx)
        reverse = f(target, data, rx, fc, dr, d0=d0, rx_pos=tx)
        torch.testing.assert_close(reverse, expected)
        keep = torch.tensor([0, 1, 4, 9, 10, 17, 28], device=self.device)
        sparse = f(target, data[keep], tx[keep], fc, dr, d0=d0, rx_pos=rx[keep])
        torch.testing.assert_close(sparse, expected[:, keep])

    def test_monostatic_limit(self):
        _, tx, _, target, data, fc, dr, d0 = point_scene(self.device)
        data = torch.ones_like(data)
        f = torchbp.ops.gpga_backprojection_2d_core
        mono = f(target, data, tx, fc, dr, d0=d0)
        same_position = f(target, data, tx, fc, dr, d0=d0, rx_pos=tx)
        # Existing monostatic kernels use float32 geometry/phase; the new
        # kernels evaluate the two distances and carrier in double precision.
        torch.testing.assert_close(same_position, mono, rtol=3e-3, atol=3e-3)

    def test_unsupported_factorization_is_explicit(self):
        _, tx, rx, _, data, fc, dr, d0 = point_scene(self.device)
        grid = {"x": (39.0, 41.0), "y": (3.0, 5.0), "nx": 4, "ny": 4}
        with self.assertRaisesRegex(ValueError, "algorithm='bp'"):
            torchbp.autofocus.gpga_tde(None, data, tx, fc, dr, grid, 1, 1,
                                      d0=d0, rx_pos=rx, algorithm="cfbp")

    def test_tde_reduces_injected_bistatic_range_error(self):
        t, tx, rx, target, _, fc, dr, d0 = point_scene(self.device)
        error = torch.zeros_like(tx)
        error[:, 0] = (0.003*torch.sin(4*torch.pi*t)).float()
        true_tx, true_rx = tx+error, rx+error
        def path(a, b):
            return (torch.linalg.vector_norm(target.double()-a.double(), dim=1) +
                    torch.linalg.vector_norm(target.double()-b.double(), dim=1))
        true_path = path(true_tx, true_rx)
        r = -d0 + dr*torch.arange(800, device=self.device, dtype=torch.float64)
        data = (torch.exp(-0.5*((r[None]-true_path[:, None]/2)/0.2)**2) *
                torch.exp(-2j*torch.pi*fc*true_path[:, None]/C0)).to(torch.complex64)
        grid = {"x": (40., 40.5), "y": (4., 4.5), "nx": 1, "ny": 1}
        rms_history = []
        _, refined = torchbp.autofocus.gpga_tde(
            None, data, tx, fc, dr, grid, 1, 1, rx_pos=rx, sweep_times=t,
            d0=d0, max_iters=1, rms_error_limit=0, remove_trend=False,
            estimate_z=False, isolation_db=0, rms_history=rms_history)
        self.assertEqual(len(rms_history), 1)
        self.assertTrue(math.isfinite(rms_history[0]))
        self.assertGreater(rms_history[0], 0)
        corrected_rx = rx+(refined-tx)
        coordinates = _sweep_coordinates(t, tx)
        before = _detrend_history(path(tx, rx)-true_path, None, coordinates)
        after = _detrend_history(path(refined, corrected_rx)-true_path, None, coordinates)
        self.assertLess(after.square().mean().sqrt().item(),
                        0.2*before.square().mean().sqrt().item())


class TestBistaticCpu(BistaticChecks, unittest.TestCase):
    pass


@unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
class TestBistaticCuda(BistaticChecks, unittest.TestCase):
    device = "cuda"


class TestBistaticGeometry(unittest.TestCase):
    def test_position_solve_uses_both_look_directions(self):
        tx = torch.tensor([[1., 2., 5.], [2., -1., 4.]], dtype=torch.float64)
        rx = torch.tensor([[5., 1., 2.], [4., 3., 2.]], dtype=torch.float64)
        targets = torch.tensor([[20., 7., 0.], [12., -9., 0.]], dtype=torch.float64)
        jac = _bistatic_look_vectors(targets, tx, rx)
        def equivalent_range(t, r):
            return (torch.linalg.vector_norm(targets[None]-t[:, None], dim=-1) +
                    torch.linalg.vector_norm(targets[None]-r[:, None], dim=-1))/2
        step = 1e-5
        for d in range(3):
            delta = torch.zeros_like(tx)
            delta[:, d] = step
            derivative = (equivalent_range(tx+delta, rx+delta) -
                          equivalent_range(tx-delta, rx-delta))/(2*step)
            torch.testing.assert_close(derivative, -jac[..., d], rtol=1e-8, atol=1e-8)

    def test_bp_gradients_include_both_antennas(self):
        tx = torch.tensor([[0.13, -1.1, 5.], [0.27, 1.2, 5.2]],
                          dtype=torch.float64, requires_grad=True)
        rx = torch.tensor([[0.8, 0.3, 5.4], [1.0, 2.1, 5.7]],
                          dtype=torch.float64, requires_grad=True)
        torch.manual_seed(4)
        data = torch.randn(2, 160, dtype=torch.complex128)
        grid = {"x": (20.1, 20.2), "y": (3.3, 3.4), "nx": 1, "ny": 1}
        def focus(t, r):
            return torchbp.ops.backprojection_cart_2d(data, grid, 1e8, 0.2, t, rx_pos=r)
        self.assertTrue(torch.autograd.gradcheck(focus, (tx, rx), eps=1e-6, atol=2e-5))

    def test_detrending_uses_actual_irregular_times(self):
        t = torch.tensor([0., 0.1, 0.2, 0.9, 1.4, 2.7], dtype=torch.float64)
        pos = torch.zeros(len(t), 3)
        coordinates = _sweep_coordinates(t+1_700_000_000, pos)
        phi = 2+3*coordinates
        residual = _detrend_history(phi, None, coordinates)
        torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-12, rtol=0)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            _sweep_coordinates(torch.tensor([0., 0., 1.]), pos[:3])


if __name__ == "__main__":
    unittest.main()
