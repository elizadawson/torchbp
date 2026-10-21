#include "../bistatic_checks.h"
#include <ATen/Parallel.h>
#include <torch/library.h>

namespace torchbp {

static at::Tensor bistatic_samples_cpu(const at::Tensor& target_pos,
        const at::Tensor& data, const at::Tensor& tx_pos,
        const at::Tensor& rx_pos, double fc, double r_res, double d0,
        double data_fmod, bool sum_sweeps) {
    check_bistatic_inputs(target_pos, data, tx_pos, rx_pos, fc, r_res, d0, data_fmod);
    const auto targets = target_pos.contiguous();
    const auto samples = data.contiguous();
    const auto tx = tx_pos.contiguous();
    const auto rx = rx_pos.contiguous();
    const int64_t np = targets.size(0), ns = data.size(0), nr = data.size(1);
    auto out = at::zeros(sum_sweeps ? std::vector<int64_t>{np}
                                  : std::vector<int64_t>{np, ns}, data.options());
    const auto* p = targets.data_ptr<float>();
    const auto* t = tx.data_ptr<float>();
    const auto* r = rx.data_ptr<float>();
    const auto* s = samples.data_ptr<c10::complex<float>>();
    auto* o = out.data_ptr<c10::complex<float>>();
    constexpr double tau = 6.283185307179586476925286766559;
    const double k = tau * fc / 299792458.0;
    at::parallel_for(0, np, 1, [&](int64_t begin, int64_t end) {
        for (int64_t j = begin; j < end; ++j) {
            double re = 0, im = 0;
            for (int64_t n = 0; n < ns; ++n) {
                double rt2 = 0, rr2 = 0;
                for (int d = 0; d < 3; ++d) {
                    const double dt = double(p[3*j+d]) - t[3*n+d];
                    const double dr = double(p[3*j+d]) - r[3*n+d];
                    rt2 += dt*dt; rr2 += dr*dr;
                }
                const double path = std::sqrt(rt2) + std::sqrt(rr2);
                const double u = (0.5*path + d0) / r_res;
                if (!std::isfinite(u) || u < 0 || u >= nr-1) continue;
                const int64_t lo = static_cast<int64_t>(u);
                const double f = u-lo;
                const auto a = s[n*nr+lo], b = s[n*nr+lo+1];
                const double vr = a.real() + f*(b.real()-a.real());
                const double vi = a.imag() + f*(b.imag()-a.imag());
                const double phase = k*path-data_fmod*u;
                const double cs = std::cos(phase), sn = std::sin(phase);
                const double yr = vr*cs-vi*sn, yi = vr*sn+vi*cs;
                if (sum_sweeps) { re += yr; im += yi; }
                else o[j*ns+n] = c10::complex<float>(yr, yi);
            }
            if (sum_sweeps) o[j] = c10::complex<float>(re, im);
        }
    });
    return out;
}

TORCH_LIBRARY_FRAGMENT(torchbp, m) {
    m.def("bistatic_samples(Tensor target_pos, Tensor data, Tensor tx_pos, Tensor rx_pos, float fc, float r_res, float d0, float data_fmod, bool sum_sweeps) -> Tensor");
}
TORCH_LIBRARY_IMPL(torchbp, CPU, m) {
    m.impl("bistatic_samples", &bistatic_samples_cpu);
}
} // namespace torchbp
