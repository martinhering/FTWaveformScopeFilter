//
//  ScopeWaveform.metal
//  Colorcast
//
//  Created by Martin Hering on 16.01.18.
//  Copyright © 2018 Vemedio. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

float scope_waveform_noisy(float2 co) {
    float2 seed = float2(sin(co.x), cos(co.y));
    return (fract(sin(dot(seed ,float2(12.9898,78.233))) * 43758.5453)-0.5) * 0.005;
}

float4 scope_waveform_add_axis(uint y, ushort hmax, float4 out) {
    float4 lineColor = float4(0.5, 0.4, 0.0, 1.0);
    for(uint i=0; i<=10; i++) {
        float lineY = floor((float)(i*hmax/10));
        float mix = clamp(fabs(lineY - (float)y), 0.0, 1.0);
        
        out.rgb = out.rgb*mix + lineColor.rgb*(1.0-mix);
    }
    
    return out;
}

kernel void
scope_waveform_compute(texture2d<float, access::sample>       inTexture         [[texture(0)]],
                       volatile device atomic_uint*           columnDataRed     [[buffer(0)]],
                       volatile device atomic_uint*           columnDataGreen   [[buffer(1)]],
                       volatile device atomic_uint*           columnDataBlue    [[buffer(2)]],
                       sampler                                wrapSampler       [[sampler(0)]],
                       uint2                                  gid               [[thread_position_in_grid]])
{
    // Check if the pixel is within the bounds of the output texture
    if((gid.x >= inTexture.get_width()) || (gid.y >= inTexture.get_height())) {
        return;
    }
    
    ushort w = inTexture.get_width();
    ushort h = inTexture.get_height();
    ushort hmax = h-1;
    float noise = scope_waveform_noisy(float2(gid.x, gid.y));
    
    if (gid.x > 0 && gid.x < w-1) {
        float4 srcPx  = inTexture.sample(wrapSampler, float2(gid));
        
        ushort y = (ushort)(clamp(0.0, (float)((srcPx.r+noise) * hmax), (float)hmax));
        atomic_fetch_add_explicit(columnDataRed + ((y * w) + gid.x), 1, memory_order_relaxed);

        y = (ushort)(clamp(0.0, (float)((srcPx.g+noise) * hmax), (float)hmax));
        atomic_fetch_add_explicit(columnDataGreen + ((y * w) + gid.x), 1, memory_order_relaxed);
        
        y = (ushort)(clamp(0.0, (float)((srcPx.b+noise) * hmax), (float)hmax));
        atomic_fetch_add_explicit(columnDataBlue + ((y * w) + gid.x), 1, memory_order_relaxed);
    }
}


kernel void
scope_waveform_blend(texture2d<float, access::sample>       inTexture       [[texture(0)]],
                     texture2d<float, access::write>        outTexture      [[texture(1)]],
                     volatile device atomic_uint*           columnDataRed   [[buffer(0)]],
                     volatile device atomic_uint*           columnDataGreen [[buffer(1)]],
                     volatile device atomic_uint*           columnDataBlue  [[buffer(2)]],
                     sampler                                wrapSampler     [[sampler(0)]],
                     uint2                                  gid             [[thread_position_in_grid]])
{
    ushort w = inTexture.get_width();
    ushort h = inTexture.get_height();
    ushort hmax = h-1;

    uint y = (uint)(clamp((float)(hmax-gid.y), 0.0, (float)hmax));
    uint cid = (y * w) + gid.x;
    uint red = atomic_load_explicit( columnDataRed + cid, memory_order_relaxed );
    uint green = atomic_load_explicit( columnDataGreen + cid, memory_order_relaxed );
    uint blue = atomic_load_explicit( columnDataBlue + cid, memory_order_relaxed );
    
    float4 out = float4(clamp(red / 5.0, 0.0, 1.0),
                        clamp(green / 5.0, 0.0, 1.0),
                        clamp(blue / 5.0, 0.0, 1.0),
                        1.0);
    
    out = scope_waveform_add_axis(y, hmax, out);
    
    outTexture.write(out, gid);
}

kernel void
scope_waveform_luma(texture2d<float, access::sample>        inTexture       [[texture(0)]],
                    texture2d<float, access::write>         outTexture      [[texture(1)]],
                    volatile device atomic_uint*            columnDataRed   [[buffer(0)]],
                    volatile device atomic_uint*            columnDataGreen [[buffer(1)]],
                    volatile device atomic_uint*            columnDataBlue  [[buffer(2)]],
                    sampler                                 wrapSampler     [[sampler(0)]],
                    uint2                                   gid             [[thread_position_in_grid]])
{
    ushort w = inTexture.get_width();
    ushort h = inTexture.get_height();
    ushort hmax = h-1;
    
    uint y = (uint)(clamp((float)(hmax-gid.y), 0.0, (float)hmax));
    uint cid = (y * w) + gid.x;
    uint red = atomic_load_explicit( columnDataRed + cid, memory_order_relaxed );
    uint green = atomic_load_explicit( columnDataGreen + cid, memory_order_relaxed );
    uint blue = atomic_load_explicit( columnDataBlue + cid, memory_order_relaxed );
    
    float4 out = float4(clamp(red / 5.0, 0.0, 1.0),
                        clamp(green / 5.0, 0.0, 1.0),
                        clamp(blue / 5.0, 0.0, 1.0),
                        1.0);
    
    float luma = 0.299 * out.r + 0.587 * out.g + 0.114 * out.b;
    out = float4(luma, luma, luma, 1.0);
    
    out = scope_waveform_add_axis(y, hmax, out);
    
    outTexture.write(out, gid);
}


kernel void
scope_waveform_parade(texture2d<float, access::sample>      inTexture       [[texture(0)]],
                      texture2d<float, access::write>       outTexture      [[texture(1)]],
                      volatile device atomic_uint*          columnDataRed   [[buffer(0)]],
                      volatile device atomic_uint*          columnDataGreen [[buffer(1)]],
                      volatile device atomic_uint*          columnDataBlue  [[buffer(2)]],
                      sampler                               wrapSampler     [[sampler(0)]],
                      uint2                                 gid             [[thread_position_in_grid]])
{
    ushort w = inTexture.get_width();
    ushort h = inTexture.get_height();
    ushort hmax = h-1;
    ushort channel = (ushort)floor(((float)gid.x / w * 3));
    
    uint y = (uint)(clamp((float)(hmax-gid.y), 0.0, (float)hmax));
    uint x = fmod((float)(gid.x * 3), w);
    uint cid = (y * w) + x;
    uint red = atomic_load_explicit( columnDataRed + cid, memory_order_relaxed );
    uint green = atomic_load_explicit( columnDataGreen + cid, memory_order_relaxed );
    uint blue = atomic_load_explicit( columnDataBlue + cid, memory_order_relaxed );
    
    
    float4 out = {0.0, 0.0, 0.0, 1.0};
    
    if (channel == 0) { out.r = clamp(red / 5.0, 0.0, 1.0); } else { out.r = 0.0; }
    if (channel == 1) { out.g = clamp(green / 5.0, 0.0, 1.0); } else { out.g = 0.0; }
    if (channel == 2) { out.b = clamp(blue / 5.0, 0.0, 1.0); } else { out.b = 0.0; }
    
    out = scope_waveform_add_axis(y, hmax, out);

    outTexture.write(out, gid);
}
