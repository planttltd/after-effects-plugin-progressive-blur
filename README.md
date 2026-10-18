# After Effects Plugin: Progressive Blur

A GPU-accelerated (Metal) After Effects effect for Apple Silicon. Linear progressive blur: radius interpolates from **Point 1** to **Point 2** along the line between them (clamped beyond each point). Metal GPU render, plus a CPU fallback for 8/16/32 bpc.

| File | What |
|---|---|
| `ProgressiveBlur.mm` | AE glue: params, SmartFX pre-render/render, GPU device setup |
| `FPB_Core.h` | Shared math, CPU blur (GCD + `simd`), embedded Metal shader source |
| `FPB_Metal.h` | Pipeline build + command encoding (used by plugin and test) |
| `ProgressiveBlurPiPL.r` | PiPL resource (flags match `GlobalSetup`) |
| `test_fpb.mm` | Self-check: GPU vs CPU, zero-radius identity, flat field, UHD timing |
| `build.sh` | Builds and signs `build/ProgressiveBlur.plugin` |

## Build

Requires Xcode (for `clang`, `Rez`, Metal) and the Adobe After Effects SDK. Download the SDK from developer.adobe.com/after-effects.

```bash
AE_SDK=~/AfterEffectsSDK/Examples ./build.sh
```

```bash
AE_SDK=~/AfterEffectsSDK/Examples ./build.sh --install
```

- `--install` copies the plugin to `/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore`. Restart AE afterwards.
- The effect appears under **Effect › Blur & Sharpen › Progressive Blur**.
- GPU is used when **Project Settings › Video Rendering and Effects** is set to *Mercury GPU Acceleration (Metal)*. Otherwise AE calls the CPU path automatically.

## How it renders

- **Radius per pixel:** `R = lerp(R1, R2, t)`, where `t` = the pixel's projection onto P1→P2, clamped to 0–1. σ = R/2 (the common UI-design blur convention).
- **Gaussian approximation:** 3 box passes per axis. Each box is **O(1) per pixel regardless of radius**: a prefix sum of the line, then two lookups.
- **Fractional radii:** the box edges are lerped, so the gradient is smooth with no stepping bands.
- **GPU:** one threadgroup per line. The prefix sum uses SIMD-group scans (`simd_prefix_inclusive_sum`), and the box evaluation runs in the same dispatch. Scratch buffers are `StorageModePrivate`, and the kernels are compiled at `GPU_DEVICE_SETUP` from embedded source.
- **Alpha:** blurs premultiplied (premultiply → blur → unpremultiply), so transparent edges don't halo.
- **Measured on this Mac, UHD, r = 250:** GPU kernels 14 ms; CPU 72 ms.

## Known limits (`ponytail:` comments in source)

Built and verified against After Effects SDK 26.5 (macOS).


- Non-uniform downsample or non-square pixel aspect ratio uses a single averaged radius.
- Float prefix sums carry about 1e-4 noise at very long lines. It is invisible at 8/16 bpc and only applies where blur > 0, because radius 0 is copied exactly.
