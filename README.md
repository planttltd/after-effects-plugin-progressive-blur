# After Effects Plugin: Progressive Blur

A **Figma-style progressive blur** for After Effects: a variable blur that ramps from sharp to soft along a line, the way the progressive blur works in Figma and other UI-design tools. It's GPU-accelerated with Metal on Apple Silicon and has a CPU fallback for 8/16/32 bpc.

## Controls

| Control | What |
|---|---|
| Point 1 / Point 2 | The ends of the blur gradient. The radius ramps along the line between them and holds constant beyond each point. |
| Point 1 / Point 2 Blur Radius | Blur strength at each point, from 0 to 250. |
| Curve | How the radius ramps: Linear, Ease In, Ease Out, Ease In Out or Exponential. |
| Flip Positions | Swaps Point 1 and Point 2. |
| Flip Amounts | Swaps the two radii. |

The blur spreads past the layer's bounds the same way AE's built-in blurs do.

## Files

| File | What |
|---|---|
| `ProgressiveBlur.mm` | AE glue: params, SmartFX pre-render/render, GPU device setup |
| `FPB_Core.h` | Shared math, CPU blur (GCD + `simd`), embedded Metal shader source |
| `FPB_Metal.h` | Pipeline build + command encoding (used by plugin and test) |
| `ProgressiveBlurPiPL.r` | PiPL resource (flags match `GlobalSetup`) |
| `test_fpb.mm` | Self-check: GPU vs CPU for every curve, zero-radius identity, flat field, spill past layer edge, UHD timing |
| `build.sh` | Builds and signs `build/ProgressiveBlur.plugin` |

## Build

Requires Xcode (for `clang`, `Rez`, Metal) and the Adobe After Effects SDK. Download the SDK from developer.adobe.com/after-effects.

```bash
AE_SDK=~/AfterEffectsSDK/Examples ./build.sh
```

```bash
AE_SDK=~/AfterEffectsSDK/Examples ./build.sh --install
```

- `--install` copies the plugin into the newest `/Applications/Adobe After Effects */Plug-ins` folder (no sudo). Restart AE afterwards.
- The effect appears under **Effect › Blur & Sharpen › Progressive Blur**.
- GPU is used when **Project Settings › Video Rendering and Effects** is set to *Mercury GPU Acceleration (Metal)*. Otherwise AE calls the CPU path automatically.

## How it renders

- **Radius per pixel:** `R = lerp(R1, R2, curve(t))`, where `t` = the pixel's projection onto P1→P2, clamped to 0–1. σ = R/2, the same blur-value convention Figma uses.
- **Beyond the layer bounds:** the effect sets `PF_OutFlag_I_EXPAND_BUFFER` and grows its result rect by the blur's reach. It blurs inside a working buffer padded with transparent pixels.
- **Gaussian approximation:** 3 box passes per axis. Each box is **O(1) per pixel regardless of radius**: a prefix sum of the line, then two lookups.
- **Fractional radii:** the box edges are lerped, so the gradient is smooth with no stepping bands.
- **GPU:** one threadgroup per line. The prefix sum uses SIMD-group scans (`simd_prefix_inclusive_sum`), and the box evaluation runs in the same dispatch. Scratch buffers are `StorageModePrivate`, and the kernels are compiled at `GPU_DEVICE_SETUP` from embedded source.
- **Alpha:** blurs premultiplied (premultiply → blur → unpremultiply), so transparent edges don't halo.
- **Measured on this Mac, UHD, r = 250:** GPU kernels 14 ms; CPU 71 ms.

## Known limits (`ponytail:` comments in source)

Built and verified against After Effects SDK 26.5 (macOS).

- Non-uniform downsample or non-square pixel aspect ratio uses a single averaged radius.
- Flip buttons swap the values at the current time. If those properties are keyframed, AE adds a keyframe rather than flipping the whole animation.
- Float prefix sums carry about 1e-4 noise at very long lines. It is invisible at 8/16 bpc and only applies where blur > 0, because radius 0 is copied exactly.

---

Figma is a trademark of Figma, Inc. This project is not affiliated with or endorsed by Figma.
