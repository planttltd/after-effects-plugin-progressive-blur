// FPB_Metal.h — Metal pipeline build + encode. Shared by the plugin and test_fpb.mm.
// Compile as Objective-C++ with ARC (-fobjc-arc).
#pragma once
#import <Metal/Metal.h>
#include "FPB_Core.h"

struct FPBMetal {
    id<MTLComputePipelineState> load, pass, store;
};

// Returns nil-free pipelines or an NSError.
inline NSError* fpb_metal_build(id<MTLDevice> device, FPBMetal& out) {
    NSError* err = nil;
    // default compile options = fast math
    id<MTLLibrary> lib = [device newLibraryWithSource:@(kFPBMetalSource) options:nil error:&err];
    if (!lib) return err;   // err may be set for mere warnings; nil lib is the real signal
    auto make = [&](NSString* name) -> id<MTLComputePipelineState> {
        id<MTLFunction> fn = [lib newFunctionWithName:name];
        return fn ? [device newComputePipelineStateWithFunction:fn error:&err] : nil;
    };
    out.load = make(@"fpb_load"); out.pass = make(@"fpb_pass"); out.store = make(@"fpb_store");
    if (!out.load || !out.pass || !out.store)
        return err ?: [NSError errorWithDomain:@"FPB" code:1 userInfo:nil];
    return nil;
}

// Encodes load -> 2*FPB_ITERATIONS box passes -> store. src/dst are AE's GPU
// world buffers (BGRA float4, pitch in p.srcPitch/p.dstPitch). Scratch lives in
// GPU-private memory; the command buffer retains it until completion.
inline bool fpb_metal_encode(const FPBMetal& m, id<MTLDevice> device, id<MTLCommandBuffer> cb,
                             id<MTLBuffer> src, id<MTLBuffer> dst, FPBParams p) {
    const NSUInteger px = (NSUInteger)p.width * p.height;
    const NSUInteger prefixLen = (NSUInteger)p.height * (p.width + 1) > (NSUInteger)p.width * (p.height + 1)
                               ? (NSUInteger)p.height * (p.width + 1) : (NSUInteger)p.width * (p.height + 1);
    id<MTLBuffer> a = [device newBufferWithLength:px * 16 options:MTLResourceStorageModePrivate];
    id<MTLBuffer> b = [device newBufferWithLength:px * 16 options:MTLResourceStorageModePrivate];
    id<MTLBuffer> P = [device newBufferWithLength:prefixLen * 16 options:MTLResourceStorageModePrivate];
    if (!a || !b || !P) return false;

    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    auto grid2D = [&](id<MTLComputePipelineState> ps, int w, int h) {
        NSUInteger tw = ps.threadExecutionWidth, th = ps.maxTotalThreadsPerThreadgroup / tw;
        [enc setComputePipelineState:ps];
        [enc dispatchThreads:MTLSizeMake(w, h, 1) threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
    };

    [enc setBuffer:(src ?: a) offset:0 atIndex:0];   // src nil = empty input layer
    [enc setBuffer:a offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof p atIndex:2];
    grid2D(m.load, p.width, p.height);

    // 1024 threads/line max (sgSums[32] assumes <= 32 SIMD groups).
    const NSUInteger tg = MIN((NSUInteger)1024, m.pass.maxTotalThreadsPerThreadgroup);
    [enc setComputePipelineState:m.pass];
    [enc setBuffer:P offset:0 atIndex:1];
    for (int k = 0; k < 2 * FPB_ITERATIONS; k++) {
        p.axis = k & 1;
        [enc setBuffer:(k & 1) ? b : a offset:0 atIndex:0];
        [enc setBuffer:(k & 1) ? a : b offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof p atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(p.axis ? p.width : p.height, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    }

    [enc setBuffer:a offset:0 atIndex:0];
    [enc setBuffer:dst offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof p atIndex:2];
    grid2D(m.store, p.outW, p.outH);
    [enc endEncoding];
    return true;
}
