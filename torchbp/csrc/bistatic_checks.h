#pragma once

#include <torch/all.h>
#include <cmath>

namespace torchbp {
inline void check_bistatic_inputs(const at::Tensor& targets,
        const at::Tensor& data, const at::Tensor& tx, const at::Tensor& rx,
        double fc, double r_res, double d0, double data_fmod) {
    TORCH_CHECK(data.dim() == 2 && data.size(0) > 0 && data.size(1) >= 2,
                "data must have shape [sweeps, range bins], with at least two bins");
    TORCH_CHECK(targets.dim() == 2 && targets.size(1) == 3,
                "target_pos must have shape [points, 3]");
    TORCH_CHECK(tx.dim() == 2 && tx.size(0) == data.size(0) && tx.size(1) == 3,
                "tx_pos must have shape [sweeps, 3]");
    TORCH_CHECK(rx.sizes() == tx.sizes(), "rx_pos must match tx_pos");
    TORCH_CHECK(data.scalar_type() == at::kComplexFloat &&
                targets.scalar_type() == at::kFloat &&
                tx.scalar_type() == at::kFloat && rx.scalar_type() == at::kFloat,
                "compiled bistatic kernels require complex64 data and float32 positions");
    TORCH_CHECK(targets.device() == data.device() && tx.device() == data.device()
                && rx.device() == data.device(), "all tensors must share a device");
    TORCH_CHECK(std::isfinite(fc) && fc > 0 && std::isfinite(r_res) && r_res > 0
                && std::isfinite(d0) && std::isfinite(data_fmod),
                "invalid frequency, range spacing, range origin, or modulation");
}
} // namespace torchbp
