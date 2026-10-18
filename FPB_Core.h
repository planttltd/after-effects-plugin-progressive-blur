// FPB_Core.h — Progressive Blur: shared math, CPU path and Metal source.
// AE-free so the self-check (test_fpb.mm) can compare CPU vs GPU directly.
//
// Algorithm (identical on CPU and GPU):
//   per-pixel radius R = lerp(R1, R2, curve(t)), t = clamp(proj of pixel onto P1->P2)
//   sigma = R / 2   (blur value = 2 * Gaussian sigma, common UI-design convention)
//   Gaussian ~= 3 box blurs per axis (H,V,H,V,H,V). Each box is O(1)/pixel
//   regardless of radius via a 1-D prefix sum of the line, and supports a
//   fractional radius so the gradient is smooth (no banding between steps).
#pragma once
#include <simd/simd.h>
#include <cmath>
#include <vector>
#include <dispatch/dispatch.h>

enum { FPB_ITERATIONS = 3 };   // box passes per axis
enum { FPB_CURVE_LINEAR = 0, FPB_CURVE_EASE_IN, FPB_CURVE_EASE_OUT, FPB_CURVE_EASE_IN_OUT, FPB_CURVE_EXPONENTIAL };

// Layout mirrored in the Metal source below (all 4-byte scalars).
struct FPBParams {
    int   width, height;      // working buffer size (output rect + blur padding)
    int   srcPitch, dstPitch; // pixels per row of AE source / AE dest
    int   axis;               // 0 = horizontal, 1 = vertical
    int   inOffX, inOffY;     // AE input (0,0) inside working buffer (may be <0)
    int   inW, inH;           // AE input size; outside it is transparent
    int   outOffX, outOffY;   // AE output (0,0) inside working buffer
    int   outW, outH;
    int   premultiply;        // 1 = AE pixels are straight alpha
    int   curve;              // FPB_CURVE_*
    float originX, originY;   // layer coords of working buffer (0,0)
    float p1x, p1y, p2x, p2y; // points, layer coords (downsampled)
    float r1, r2;             // radii, downsampled px
};

// Cubic easings; exponential = (2^10t - 1) / 1023. All map 0->0, 1->1.
inline float fpb_curve(int c, float t) {
    float u = 1.f - t;
    switch (c) {
        case FPB_CURVE_EASE_IN:     return t * t * t;
        case FPB_CURVE_EASE_OUT:    return 1.f - u * u * u;
        case FPB_CURVE_EASE_IN_OUT: return t < 0.5f ? 4.f * t * t * t : 1.f - 4.f * u * u * u;
        case FPB_CURVE_EXPONENTIAL: return (exp2f(10.f * t) - 1.f) / 1023.f;
        default:                    return t;
    }
}

// Box radius for one of FPB_ITERATIONS passes. 3 boxes of radius r have
// variance r(r+1) -> solve for the target sigma.
inline float fpb_box_radius(const FPBParams& p, float x, float y) {
    float dx = p.p2x - p.p1x, dy = p.p2y - p.p1y, dd = dx * dx + dy * dy;
    float t = dd > 1e-6f ? ((x - p.p1x) * dx + (y - p.p1y) * dy) / dd : 0.f;
    t = fpb_curve(p.curve, fminf(fmaxf(t, 0.f), 1.f));
    float sigma = 0.5f * (p.r1 + (p.r2 - p.r1) * t);
    return sqrtf(sigma * sigma + 0.25f) - 0.5f;
}

// P = exclusive prefix of the line (len+1 entries). Fractional box:
// lerp between box(ri) and box(ri+1) in sum space, normalised by its weight.
inline simd_float4 fpb_box(const simd_float4* P, int len, int i, float r) {
    int ri = (int)r; float f = r - ri;
    auto at = [&](int k) { return P[k < 0 ? 0 : (k > len ? len : k)]; };
    simd_float4 inner = at(i + ri + 1) - at(i - ri);
    simd_float4 outer = at(i + ri + 2) - at(i - ri - 1);
    return (inner * (1.f - f) + outer * f) / ((2 * ri + 1) + 2.f * f);
}

// One box pass along p.axis, src -> dst (both width*height, tightly packed).
// Lines run in parallel on GCD (P+E cores); outside the buffer counts as transparent.
inline void fpb_cpu_pass(const FPBParams& p, const simd_float4* src, simd_float4* dst) {
    const bool h = p.axis == 0;
    const int len = h ? p.width : p.height, lines = h ? p.height : p.width;
    const int step = h ? 1 : p.width;
    dispatch_apply((size_t)lines, DISPATCH_APPLY_AUTO, ^(size_t line) {
        // ponytail: per-line heap alloc, negligible vs the line's work
        std::vector<simd_float4> P(len + 1);
        const int base = h ? (int)line * p.width : (int)line;
        simd_float4 acc = 0; P[0] = acc;
        for (int i = 0; i < len; i++) { acc += src[base + i * step]; P[i + 1] = acc; }
        for (int i = 0; i < len; i++) {
            float x = h ? i : (float)line, y = h ? (float)line : i;
            float r = fpb_box_radius(p, x + p.originX, y + p.originY);
            // r~0: copy exactly (prefix-sum differences carry float cancellation noise)
            dst[base + i * step] = r < 1e-3f ? src[base + i * step] : fpb_box(P.data(), len, i, r);
        }
    });
}

// Full blur on a premultiplied working buffer, in place (tmp is scratch, same size).
inline void fpb_cpu_blur(FPBParams p, simd_float4* buf, simd_float4* tmp) {
    for (int k = 0; k < FPB_ITERATIONS; k++) {
        p.axis = 0; fpb_cpu_pass(p, buf, tmp);
        p.axis = 1; fpb_cpu_pass(p, tmp, buf);
    }
}

// ---------------------------------------------------------------------------
// Metal (compiled at GPU_DEVICE_SETUP with newLibraryWithSource).
// fpb_pass: one threadgroup per line. The line's prefix sum is built with
// SIMD-group scans (simd_prefix_inclusive_sum) + a threadgroup scan of the
// SIMD totals, chunk by chunk, then the same group evaluates the O(1) box.
// GPU worlds are BGRA float4: alpha is .w.
static const char* kFPBMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct FPBParams {
    int width, height, srcPitch, dstPitch, axis, inOffX, inOffY, inW, inH,
        outOffX, outOffY, outW, outH, premultiply, curve;
    float originX, originY, p1x, p1y, p2x, p2y, r1, r2;
};

static float fpb_curve(int c, float t) {
    float u = 1.0f - t;
    switch (c) {
        case 1:  return t * t * t;
        case 2:  return 1.0f - u * u * u;
        case 3:  return t < 0.5f ? 4.0f * t * t * t : 1.0f - 4.0f * u * u * u;
        case 4:  return (exp2(10.0f * t) - 1.0f) / 1023.0f;
        default: return t;
    }
}

static float fpb_box_radius(constant FPBParams& p, float2 pos) {
    float2 a = float2(p.p1x, p.p1y), d = float2(p.p2x, p.p2y) - a;
    float dd = dot(d, d);
    float t = fpb_curve(p.curve, dd > 1e-6f ? saturate(dot(pos - a, d) / dd) : 0.0f);
    float sigma = 0.5f * mix(p.r1, p.r2, t);
    return sqrt(sigma * sigma + 0.25f) - 0.5f;
}

static float4 fpb_box(device const float4* P, int len, int i, float r) {
    int ri = int(r); float f = r - float(ri);
    float4 inner = P[clamp(i + ri + 1, 0, len)] - P[clamp(i - ri,     0, len)];
    float4 outer = P[clamp(i + ri + 2, 0, len)] - P[clamp(i - ri - 1, 0, len)];
    return (inner * (1.0f - f) + outer * f) / (float(2 * ri + 1) + 2.0f * f);
}

kernel void fpb_load(device const float4* src [[buffer(0)]],
                     device float4* dst       [[buffer(1)]],
                     constant FPBParams& p    [[buffer(2)]],
                     uint2 g [[thread_position_in_grid]])
{
    if (int(g.x) >= p.width || int(g.y) >= p.height) return;
    int sx = int(g.x) - p.inOffX, sy = int(g.y) - p.inOffY;
    float4 c = 0.0f;   // outside the layer = transparent, so the blur spills past its bounds
    if (sx >= 0 && sy >= 0 && sx < p.inW && sy < p.inH) c = src[sy * p.srcPitch + sx];
    dst[g.y * p.width + g.x] = p.premultiply ? float4(c.xyz * c.w, c.w) : c;
}

kernel void fpb_pass(device const float4* src [[buffer(0)]],
                     device float4* prefix    [[buffer(1)]],
                     device float4* dst       [[buffer(2)]],
                     constant FPBParams& p    [[buffer(3)]],
                     uint line   [[threadgroup_position_in_grid]],
                     uint tid    [[thread_index_in_threadgroup]],
                     uint tgSize [[threads_per_threadgroup]],
                     uint lane   [[thread_index_in_simdgroup]],
                     uint sg     [[simdgroup_index_in_threadgroup]],
                     uint sgSize [[threads_per_simdgroup]],
                     uint sgN    [[simdgroups_per_threadgroup]])
{
    threadgroup float4 sgSums[32];
    threadgroup float4 chunkTotal;

    const bool h = p.axis == 0;
    const int len = h ? p.width : p.height;
    const int step = h ? 1 : p.width;
    const int base = h ? int(line) * p.width : int(line);
    device float4* P = prefix + line * uint(len + 1);

    if (tid == 0) P[0] = 0.0f;
    float4 running = 0.0f;
    for (int start = 0; start < len; start += int(tgSize)) {
        int i = start + int(tid);
        float4 v = i < len ? src[base + i * step] : 0.0f;
        float4 incl = simd_prefix_inclusive_sum(v);
        if (lane == sgSize - 1) sgSums[sg] = incl;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            float4 s = lane < sgN ? sgSums[lane] : 0.0f;
            float4 ex = simd_prefix_exclusive_sum(s);
            float4 tot = simd_sum(s);
            if (lane < sgN) sgSums[lane] = ex;
            if (lane == 0) chunkTotal = tot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (i < len) P[i + 1] = running + sgSums[sg] + incl;
        running += chunkTotal;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup_barrier(mem_flags::mem_device);

    for (int i = int(tid); i < len; i += int(tgSize)) {
        float2 pos = h ? float2(i, line) : float2(line, i);
        float r = fpb_box_radius(p, pos + float2(p.originX, p.originY));
        dst[base + i * step] = r < 1e-3f ? src[base + i * step] : fpb_box(P, len, i, r);
    }
}

kernel void fpb_store(device const float4* src [[buffer(0)]],
                      device float4* dst       [[buffer(1)]],
                      constant FPBParams& p    [[buffer(2)]],
                      uint2 g [[thread_position_in_grid]])
{
    if (int(g.x) >= p.outW || int(g.y) >= p.outH) return;
    float4 c = src[(g.y + p.outOffY) * p.width + g.x + p.outOffX];
    if (p.premultiply) c = c.w > 1e-6f ? float4(c.xyz / c.w, c.w) : 0.0f;
    dst[g.y * p.dstPitch + g.x] = c;
}
)METAL";
