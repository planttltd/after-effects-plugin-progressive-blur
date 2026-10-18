// Self-check: GPU (exact plugin kernels) vs CPU path, plus identity + flat-field checks.
// clang++ -std=c++17 -fobjc-arc -O2 test_fpb.mm -framework Metal -framework Foundation -o test_fpb && ./test_fpb
#include "FPB_Metal.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>

static std::vector<simd_float4> cpu_run(FPBParams p, const std::vector<simd_float4>& src) {
    std::vector<simd_float4> buf(p.width * p.height), tmp(buf.size()), out(p.outW * p.outH);
    for (int i = 0; i < p.width * p.height; i++) { simd_float4 c = src[i]; c.xyz *= c.w; buf[i] = c; }
    fpb_cpu_blur(p, buf.data(), tmp.data());
    for (int y = 0; y < p.outH; y++) for (int x = 0; x < p.outW; x++) {
        simd_float4 c = buf[(y + p.outOffY) * p.width + x + p.outOffX];
        out[y * p.outW + x] = c.w > 1e-6f ? simd_make_float4(c.xyz / c.w, c.w) : 0;
    }
    return out;
}

static double g_lastGpuMs = 0;
static std::vector<simd_float4> gpu_run(id<MTLDevice> dev, const FPBMetal& m, FPBParams p,
                                        const std::vector<simd_float4>& src, bool timed = false) {
    id<MTLBuffer> s = [dev newBufferWithBytes:src.data() length:src.size() * 16 options:MTLResourceStorageModeShared];
    id<MTLBuffer> d = [dev newBufferWithLength:p.outW * p.outH * 16 options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb = [[dev newCommandQueue] commandBuffer];
    assert(fpb_metal_encode(m, dev, cb, s, d, p));
    [cb commit]; [cb waitUntilCompleted];
    assert(!cb.error);
    if (timed) g_lastGpuMs = (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
    auto* o = (simd_float4*)d.contents;
    return std::vector<simd_float4>(o, o + p.outW * p.outH);
}

int main() {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    FPBMetal m;
    if (NSError* e = fpb_metal_build(dev, m)) { printf("%s\n", e.description.UTF8String); return 1; }

    FPBParams p{};
    p.width = 1500; p.height = 777; p.srcPitch = p.width; p.premultiply = 1;
    p.outOffX = 40; p.outOffY = 30; p.outW = 1400; p.outH = 700; p.dstPitch = p.outW;
    p.originX = -40; p.originY = -30;
    p.p1x = 100; p.p1y = 50; p.p2x = 1200; p.p2y = 650; p.r1 = 0; p.r2 = 250;

    srand(1);
    std::vector<simd_float4> img(p.width * p.height);
    for (auto& c : img) c = simd_make_float4(rand() % 1000 / 999.f, rand() % 1000 / 999.f,
                                             rand() % 1000 / 999.f, rand() % 1000 / 999.f);

    // 1) GPU matches CPU
    auto c = cpu_run(p, img), g = gpu_run(dev, m, p, img);
    float maxErr = 0;
    for (size_t i = 0; i < c.size(); i++) if (c[i].w > 0.01f) maxErr = fmaxf(maxErr, simd_reduce_max(simd_abs(c[i] - g[i])));
    printf("GPU vs CPU max abs diff: %g\n", maxErr);
    assert(maxErr < 2e-3f);

    // 2) zero radius == identity
    FPBParams z = p; z.r1 = z.r2 = 0;
    auto gi = gpu_run(dev, m, z, img);
    for (int y = 0; y < p.outH; y++) for (int x = 0; x < p.outW; x++) {
        simd_float4 a = img[(y + p.outOffY) * p.width + x + p.outOffX], b = gi[y * p.outW + x];
        a.xyz *= a.w; b.xyz *= b.w;   // compare premultiplied (unpremul of tiny alpha is ill-conditioned)
        assert(simd_reduce_max(simd_abs(a - b)) < 1e-6f);
    }

    // 3) flat opaque field stays flat away from the edges (normalisation correct)
    std::vector<simd_float4> flat(img.size(), simd_make_float4(0.25f, 0.5f, 0.75f, 1));
    FPBParams f = p; f.r1 = f.r2 = 60;
    auto gf = gpu_run(dev, m, f, flat);
    simd_float4 mid = gf[(p.outH / 2) * p.outW + p.outW / 2];
    assert(simd_reduce_max(simd_abs(mid - flat[0])) < 1e-4f);

    // timing at UHD, worst-case radius
    FPBParams u = p; u.width = 3840 + 760; u.height = 2160 + 760; u.srcPitch = u.width;
    u.outW = 3840; u.outH = 2160; u.dstPitch = u.outW; u.outOffX = u.outOffY = 380; u.r1 = u.r2 = 250;
    std::vector<simd_float4> big(u.width * u.height, simd_make_float4(0.5f));
    gpu_run(dev, m, u, big);  // warm-up
    double gpuMs = gpu_run(dev, m, u, big, true).size() ? g_lastGpuMs : 0;
    CFAbsoluteTime t1 = CFAbsoluteTimeGetCurrent(); cpu_run(u, big);
    CFAbsoluteTime t2 = CFAbsoluteTimeGetCurrent();
    printf("UHD r=250: GPU %.1f ms (kernel time), CPU %.1f ms\nall checks passed\n", gpuMs, (t2 - t1) * 1e3);
}
