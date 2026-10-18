// ProgressiveBlur.mm — After Effects SmartFX effect, Apple Silicon only.
// GPU: Metal via PF_Cmd_SMART_RENDER_GPU (Mercury GPU Acceleration).
// CPU fallback: same algorithm, GCD-parallel, 8/16/32 bpc.
// Build with -fobjc-arc (see build.sh).
#include "AEConfig.h"
#include "entry.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_EffectCBSuites.h"
#include "AE_EffectGPUSuites.h"
#include "AE_Macros.h"
#include "AEFX_SuiteHandlerTemplate.h"
#include "Param_Utils.h"
#include "FPB_Metal.h"
#include <new>

#define MAJOR_VERSION 1
#define MINOR_VERSION 0
#define BUG_VERSION   0
#define STAGE_VERSION PF_Stage_DEVELOP
#define BUILD_VERSION 1   // must match AE_Effect_Version in the PiPL (524289)

// AE effect worlds are straight (unpremultiplied) alpha: the SDK's own samples
// (Transformer, Shifter) and AE_EffectCB.h macros pass PF_MF_Alpha_STRAIGHT.
enum { FPB_INPUT_IS_STRAIGHT = 1 };

enum { FPB_INPUT = 0, FPB_P1, FPB_R1, FPB_P2, FPB_R2, FPB_NUM_PARAMS };
enum { P1_DISK_ID = 1, R1_DISK_ID, P2_DISK_ID, R2_DISK_ID };

struct FPBRenderData {           // computed in PreRender, consumed in SmartRender
    FPBParams p;                 // geometry filled here; buffer sizes filled at render
    PF_LRect inRect, outRect;    // layer coords
};

// ---------------------------------------------------------------- setup

static PF_Err GlobalSetup(PF_InData* in_data, PF_OutData* out_data) {
    out_data->my_version = PF_VERSION(MAJOR_VERSION, MINOR_VERSION, BUG_VERSION, STAGE_VERSION, BUILD_VERSION);
    out_data->out_flags  = PF_OutFlag_DEEP_COLOR_AWARE;                       // 0x02000000
    out_data->out_flags2 = PF_OutFlag2_FLOAT_COLOR_AWARE                      // 0x0A001400
                         | PF_OutFlag2_SUPPORTS_SMART_RENDER
                         | PF_OutFlag2_SUPPORTS_GPU_RENDER_F32
                         | PF_OutFlag2_SUPPORTS_THREADED_RENDERING;
    return PF_Err_NONE;
}

static PF_Err ParamsSetup(PF_InData* in_data, PF_OutData* out_data) {
    PF_Err err = PF_Err_NONE;
    PF_ParamDef def;
    AEFX_CLR_STRUCT(def);
    PF_ADD_POINT("Point 1", 50, 0, false, P1_DISK_ID);
    PF_ADD_FLOAT_SLIDERX("Point 1 Blur Radius", 0, 250, 0, 250, 0,
                         PF_Precision_TENTHS, PF_ValueDisplayFlag_NONE, 0, R1_DISK_ID);
    AEFX_CLR_STRUCT(def);
    PF_ADD_POINT("Point 2", 50, 100, false, P2_DISK_ID);
    PF_ADD_FLOAT_SLIDERX("Point 2 Blur Radius", 0, 250, 0, 250, 40,
                         PF_Precision_TENTHS, PF_ValueDisplayFlag_NONE, 0, R2_DISK_ID);
    out_data->num_params = FPB_NUM_PARAMS;
    return err;
}

static PF_Err GPUDeviceSetup(PF_InData* in_data, PF_OutData* out_data, PF_GPUDeviceSetupExtra* extra) {
    if (extra->input->what_gpu != PF_GPU_Framework_METAL) return PF_Err_NONE;  // CPU path then
    @autoreleasepool {
        AEFX_SuiteScoper<PF_GPUDeviceSuite1> gpu(in_data, kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1, out_data);
        PF_GPUDeviceInfo info; AEFX_CLR_STRUCT(info);
        PF_Err err = gpu->GetDeviceInfo(in_data->effect_ref, extra->input->device_index, &info);
        if (err) return err;
        auto* m = new (std::nothrow) FPBMetal;
        if (!m) return PF_Err_OUT_OF_MEMORY;
        if (NSError* e = fpb_metal_build((__bridge id<MTLDevice>)info.devicePV, *m)) {
            NSLog(@"Progressive Blur: Metal build failed: %@", e);
            delete m;
            return PF_Err_NONE;   // no GPU flag -> AE uses the CPU render
        }
        extra->output->gpu_data = m;
        out_data->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
    }
    return PF_Err_NONE;
}

static PF_Err GPUDeviceSetdown(PF_GPUDeviceSetdownExtra* extra) {
    delete static_cast<FPBMetal*>(extra->input->gpu_data);
    return PF_Err_NONE;
}

// ---------------------------------------------------------------- pre-render

static PF_LRect Intersect(PF_LRect a, const PF_LRect& b) {
    a.left = MAX(a.left, b.left); a.top = MAX(a.top, b.top);
    a.right = MIN(a.right, b.right); a.bottom = MIN(a.bottom, b.bottom);
    if (a.right < a.left) a.right = a.left;
    if (a.bottom < a.top) a.bottom = a.top;
    return a;
}

static PF_Err PreRender(PF_InData* in_data, PF_OutData* out_data, PF_PreRenderExtra* extra) {
    PF_Err err = PF_Err_NONE, err2 = PF_Err_NONE;
    PF_ParamDef p1, r1, p2, r2;
    AEFX_CLR_STRUCT(p1); AEFX_CLR_STRUCT(r1); AEFX_CLR_STRUCT(p2); AEFX_CLR_STRUCT(r2);
    const A_long t = in_data->current_time, ts = in_data->time_step;
    const A_u_long sc = in_data->time_scale;
    ERR(PF_CHECKOUT_PARAM(in_data, FPB_P1, t, ts, sc, &p1));
    ERR(PF_CHECKOUT_PARAM(in_data, FPB_R1, t, ts, sc, &r1));
    ERR(PF_CHECKOUT_PARAM(in_data, FPB_P2, t, ts, sc, &p2));
    ERR(PF_CHECKOUT_PARAM(in_data, FPB_R2, t, ts, sc, &r2));

    auto* d = err ? nullptr : new (std::nothrow) FPBRenderData{};
    if (!err && !d) err = PF_Err_OUT_OF_MEMORY;
    if (!err) {
        extra->output->pre_render_data = d;
        extra->output->delete_pre_render_data_func = [](void* v) { delete static_cast<FPBRenderData*>(v); };

        // Points arrive already scaled for downsample, in layer coords. Sliders don't.
        // ponytail: isotropic radius from the mean downsample factor; non-uniform
        // downsample / non-square PAR would need separate H and V radii.
        const float ds = 0.5f * ((float)in_data->downsample_x.num / in_data->downsample_x.den +
                                 (float)in_data->downsample_y.num / in_data->downsample_y.den);
        FPBParams& p = d->p;
        p.premultiply = FPB_INPUT_IS_STRAIGHT;
        p.p1x = FIX_2_FLOAT(p1.u.td.x_value); p.p1y = FIX_2_FLOAT(p1.u.td.y_value);
        p.p2x = FIX_2_FLOAT(p2.u.td.x_value); p.p2y = FIX_2_FLOAT(p2.u.td.y_value);
        p.r1 = (float)r1.u.fs_d.value * ds;
        p.r2 = (float)r2.u.fs_d.value * ds;

        // Pad the input request by the support of FPB_ITERATIONS boxes at max radius.
        const float sigma = 0.5f * MAX(p.r1, p.r2);
        const A_long pad = (A_long)ceilf(FPB_ITERATIONS * (sqrtf(sigma * sigma + 0.25f) + 1.5f));

        PF_RenderRequest req = extra->input->output_request;
        req.rect.left -= pad; req.rect.top -= pad; req.rect.right += pad; req.rect.bottom += pad;
        req.preserve_rgb_of_zero_alpha = FALSE;

        PF_CheckoutResult in_result;
        ERR(extra->cb->checkout_layer(in_data->effect_ref, FPB_INPUT, FPB_INPUT, &req, t, ts, sc, &in_result));
        if (!err) {
            d->inRect  = in_result.result_rect;
            d->outRect = Intersect(Intersect(extra->input->output_request.rect, in_result.max_result_rect),
                                   in_result.result_rect);
            extra->output->result_rect     = d->outRect;
            extra->output->max_result_rect = in_result.max_result_rect;
            if (extra->input->what_gpu == PF_GPU_Framework_METAL)
                extra->output->flags |= PF_RenderOutputFlag_GPU_RENDER_POSSIBLE;
        }
    }
    ERR2(PF_CHECKIN_PARAM(in_data, &p1));
    ERR2(PF_CHECKIN_PARAM(in_data, &r1));
    ERR2(PF_CHECKIN_PARAM(in_data, &p2));
    ERR2(PF_CHECKIN_PARAM(in_data, &r2));
    return err;
}

// ---------------------------------------------------------------- CPU render

template <class Px> struct Chan;
template <> struct Chan<PF_Pixel8>     { static constexpr float max = PF_MAX_CHAN8;  static constexpr bool clamp = true;  };
template <> struct Chan<PF_Pixel16>    { static constexpr float max = PF_MAX_CHAN16; static constexpr bool clamp = true;  };
template <> struct Chan<PF_PixelFloat> { static constexpr float max = 1.f;           static constexpr bool clamp = false; };

template <class Px>
static void CPURender(const FPBParams& p, const PF_EffectWorld* in, PF_EffectWorld* out) {
    std::vector<simd_float4> buf((size_t)p.width * p.height), tmp(buf.size());
    simd_float4* B = buf.data();
    const float k = 1.f / Chan<Px>::max;

    dispatch_apply((size_t)p.height, DISPATCH_APPLY_AUTO, ^(size_t y) {
        const Px* row = (const Px*)((const char*)in->data + y * in->rowbytes);
        for (int x = 0; x < p.width; x++) {
            simd_float4 c = simd_make_float4(row[x].red, row[x].green, row[x].blue, row[x].alpha) * k;
            if (p.premultiply) c.xyz *= c.w;
            B[y * p.width + x] = c;
        }
    });

    fpb_cpu_blur(p, B, tmp.data());

    dispatch_apply((size_t)p.outH, DISPATCH_APPLY_AUTO, ^(size_t y) {
        Px* row = (Px*)((char*)out->data + y * out->rowbytes);
        for (int x = 0; x < p.outW; x++) {
            simd_float4 c = B[(y + p.outOffY) * p.width + x + p.outOffX];
            if (p.premultiply) c = c.w > 1e-6f ? simd_make_float4(c.xyz / c.w, c.w) : 0;
            c *= Chan<Px>::max;
            if (Chan<Px>::clamp) c = simd_clamp(c + 0.5f, 0.f, Chan<Px>::max);
            using T = decltype(row[x].red);
            row[x].alpha = (T)c.w; row[x].red = (T)c.x; row[x].green = (T)c.y; row[x].blue = (T)c.z;
        }
    });
}

// ---------------------------------------------------------------- smart render

static PF_Err SmartRender(PF_InData* in_data, PF_OutData* out_data, PF_SmartRenderExtra* extra, bool gpu) {
    PF_Err err = PF_Err_NONE, err2 = PF_Err_NONE;
    auto* d = static_cast<FPBRenderData*>(extra->input->pre_render_data);
    PF_EffectWorld *in = nullptr, *out = nullptr;
    ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, FPB_INPUT, &in));
    ERR(extra->cb->checkout_output(in_data->effect_ref, &out));
    if (!err && d && in && out) {
        FPBParams p = d->p;
        p.width = in->width; p.height = in->height;
        p.originX = (float)d->inRect.left; p.originY = (float)d->inRect.top;
        p.outOffX = d->outRect.left - d->inRect.left;
        p.outOffY = d->outRect.top  - d->inRect.top;
        p.outW = MIN(out->width,  p.width  - p.outOffX);
        p.outH = MIN(out->height, p.height - p.outOffY);

        AEFX_SuiteScoper<PF_WorldSuite2> worlds(in_data, kPFWorldSuite, kPFWorldSuiteVersion2, out_data);
        PF_PixelFormat fmt = PF_PixelFormat_INVALID;
        ERR(worlds->PF_GetPixelFormat(in, &fmt));

        if (!err && p.outW > 0 && p.outH > 0 && p.outOffX >= 0 && p.outOffY >= 0) {
            if (gpu && fmt != PF_PixelFormat_GPU_BGRA128) {
                err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
            } else if (gpu) {
                @autoreleasepool {
                    AEFX_SuiteScoper<PF_GPUDeviceSuite1> gs(in_data, kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1, out_data);
                    PF_GPUDeviceInfo info; AEFX_CLR_STRUCT(info);
                    void *src = nullptr, *dst = nullptr;
                    ERR(gs->GetDeviceInfo(in_data->effect_ref, extra->input->device_index, &info));
                    ERR(gs->GetGPUWorldData(in_data->effect_ref, in, &src));
                    ERR(gs->GetGPUWorldData(in_data->effect_ref, out, &dst));
                    if (!err) {
                        p.srcPitch = in->rowbytes / 16; p.dstPitch = out->rowbytes / 16;  // BGRA128
                        auto* m = static_cast<const FPBMetal*>(extra->input->gpu_data);
                        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)info.command_queuePV;
                        id<MTLCommandBuffer> cb = [q commandBuffer];
                        if (!fpb_metal_encode(*m, (__bridge id<MTLDevice>)info.devicePV, cb,
                                              (__bridge id<MTLBuffer>)src, (__bridge id<MTLBuffer>)dst, p))
                            err = PF_Err_OUT_OF_MEMORY;
                        [cb commit];   // AE syncs on its own queue, like the SDK sample
                    }
                }
            } else {
                try {
                    switch (fmt) {
                        case PF_PixelFormat_ARGB32:  CPURender<PF_Pixel8>(p, in, out);     break;
                        case PF_PixelFormat_ARGB64:  CPURender<PF_Pixel16>(p, in, out);    break;
                        case PF_PixelFormat_ARGB128: CPURender<PF_PixelFloat>(p, in, out); break;
                        default: err = PF_Err_BAD_CALLBACK_PARAM;
                    }
                } catch (const std::bad_alloc&) { err = PF_Err_OUT_OF_MEMORY; }
            }
        }
    }
    ERR2(extra->cb->checkin_layer_pixels(in_data->effect_ref, FPB_INPUT));
    return err;
}

// ---------------------------------------------------------------- entry

extern "C" DllExport PF_Err PluginDataEntryFunction2(PF_PluginDataPtr inPtr, PF_PluginDataCB2 cb,
                                                     SPBasicSuite*, const char*, const char*) {
    PF_Err result = PF_Err_INVALID_CALLBACK;   // the macro assigns to `result`
    result = PF_REGISTER_EFFECT_EXT2(inPtr, cb, "Progressive Blur", "ROYP Progressive Blur",
                                     "Blur & Sharpen", AE_RESERVED_INFO, "EffectMain", "https://royplant.com");
    return result;
}

extern "C" DllExport PF_Err EffectMain(PF_Cmd cmd, PF_InData* in_data, PF_OutData* out_data,
                                       PF_ParamDef* params[], PF_LayerDef* output, void* extra) {
    try {
        switch (cmd) {
            case PF_Cmd_ABOUT:
                snprintf(out_data->return_msg, sizeof out_data->return_msg,
                         "Progressive Blur %d.%d\rLinear progressive blur, Metal accelerated.",
                         MAJOR_VERSION, MINOR_VERSION);
                return PF_Err_NONE;
            case PF_Cmd_GLOBAL_SETUP:       return GlobalSetup(in_data, out_data);
            case PF_Cmd_PARAMS_SETUP:       return ParamsSetup(in_data, out_data);
            case PF_Cmd_GPU_DEVICE_SETUP:   return GPUDeviceSetup(in_data, out_data, (PF_GPUDeviceSetupExtra*)extra);
            case PF_Cmd_GPU_DEVICE_SETDOWN: return GPUDeviceSetdown((PF_GPUDeviceSetdownExtra*)extra);
            case PF_Cmd_SMART_PRE_RENDER:   return PreRender(in_data, out_data, (PF_PreRenderExtra*)extra);
            case PF_Cmd_SMART_RENDER:       return SmartRender(in_data, out_data, (PF_SmartRenderExtra*)extra, false);
            case PF_Cmd_SMART_RENDER_GPU:   return SmartRender(in_data, out_data, (PF_SmartRenderExtra*)extra, true);
            default:                        return PF_Err_NONE;
        }
    } catch (PF_Err& e) { return e; }   // never throw into AE
}
