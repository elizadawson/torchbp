#include "../bistatic_checks.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

namespace torchbp {

template<bool SUM>
__global__ void bistatic_samples_kernel(const float* targets,
        const c10::complex<float>* data, const float* tx, const float* rx,
        c10::complex<float>* out, int64_t np, int64_t ns, int64_t nr,
        double k, double r_res, double d0, double fmod) {
    const int64_t id = int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (id >= (SUM ? np : np*ns)) return;
    const int64_t j = SUM ? id : id/ns;
    double re = 0, im = 0;
    const int64_t first = SUM ? 0 : id%ns, last = SUM ? ns : first+1;
    for (int64_t n = first; n < last; ++n) {
        double rt2 = 0, rr2 = 0;
        for (int d = 0; d < 3; ++d) {
            const double dt = double(targets[3*j+d])-tx[3*n+d];
            const double dr = double(targets[3*j+d])-rx[3*n+d];
            rt2 += dt*dt; rr2 += dr*dr;
        }
        const double path = sqrt(rt2)+sqrt(rr2);
        const double u = (0.5*path+d0)/r_res;
        if (!isfinite(u) || u < 0 || u >= nr-1) continue;
        const int64_t lo = int64_t(u);
        const double f = u-lo;
        const auto a = data[n*nr+lo], b = data[n*nr+lo+1];
        const double vr = a.real()+f*(b.real()-a.real());
        const double vi = a.imag()+f*(b.imag()-a.imag());
        double sn, cs;
        sincos(k*path-fmod*u, &sn, &cs);
        re += vr*cs-vi*sn; im += vr*sn+vi*cs;
    }
    out[id] = c10::complex<float>(re, im);
}

static at::Tensor bistatic_samples_cuda(const at::Tensor& target_pos,
        const at::Tensor& data, const at::Tensor& tx_pos,
        const at::Tensor& rx_pos, double fc, double r_res, double d0,
        double data_fmod, bool sum_sweeps) {
    check_bistatic_inputs(target_pos, data, tx_pos, rx_pos, fc, r_res, d0, data_fmod);
    const c10::cuda::CUDAGuard guard(data.device());
    const auto targets = target_pos.contiguous(), tx = tx_pos.contiguous();
    const auto samples = data.contiguous(), rx = rx_pos.contiguous();
    const int64_t np = targets.size(0), ns = data.size(0), nr = data.size(1);
    auto out = at::empty(sum_sweeps ? std::vector<int64_t>{np}
                                  : std::vector<int64_t>{np, ns}, data.options());
    const int64_t total = out.numel();
    if (!total) return out;
    const int threads = 128;
    const int64_t blocks = (total+threads-1)/threads;
    const double k = 6.283185307179586476925286766559*fc/299792458.0;
    const auto stream = at::cuda::getCurrentCUDAStream();
    const auto* p = targets.data_ptr<float>();
    const auto* s = samples.data_ptr<c10::complex<float>>();
    const auto* t = tx.data_ptr<float>();
    const auto* r = rx.data_ptr<float>();
    auto* o = out.data_ptr<c10::complex<float>>();
    if (sum_sweeps)
        bistatic_samples_kernel<true><<<blocks, threads, 0, stream>>>(
            p, s, t, r, o, np, ns, nr, k, r_res, d0, data_fmod);
    else
        bistatic_samples_kernel<false><<<blocks, threads, 0, stream>>>(
            p, s, t, r, o, np, ns, nr, k, r_res, d0, data_fmod);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

TORCH_LIBRARY_IMPL(torchbp, CUDA, m) {
    m.impl("bistatic_samples", &bistatic_samples_cuda);
}
} // namespace torchbp
