/*
 * Colorcast
 * Copyright (C) 2018 Martin Hering
 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#import "FTWaveformScopeKernel.h"
#import "FTWaveformScopeFilter.h"

#import <Metal/Metal.h>

static id<MTLSamplerState> kSamplerState;
static id<MTLDevice> kDevice;
static id<MTLComputePipelineState> kWaveformComputePipelineState;
static id<MTLComputePipelineState> kBlendComputePipelineState;
static id<MTLComputePipelineState> kParadeComputePipelineState;
static id<MTLComputePipelineState> kLumaComputePipelineState;

@implementation FTWaveformScopeKernel

+ (void) initialize {
    kDevice = MTLCreateSystemDefaultDevice();
    
    MTLSamplerDescriptor *samplerDescriptor = [MTLSamplerDescriptor new];
    samplerDescriptor.sAddressMode = MTLSamplerAddressModeClampToEdge;
    samplerDescriptor.tAddressMode = MTLSamplerAddressModeClampToEdge;
    samplerDescriptor.minFilter = MTLSamplerMinMagFilterNearest;
    samplerDescriptor.magFilter = MTLSamplerMinMagFilterNearest;
    samplerDescriptor.normalizedCoordinates = NO;
    kSamplerState = [kDevice newSamplerStateWithDescriptor:samplerDescriptor];
    
    id<MTLLibrary> defaultLibrary = [kDevice newDefaultLibrary];
    
    id<MTLFunction> computeFunction = [defaultLibrary newFunctionWithName:@"scope_waveform_compute"];
    if (computeFunction) {
        NSError* error;
        kWaveformComputePipelineState = [kDevice newComputePipelineStateWithFunction:computeFunction error:&error];
        if (error) {
            ErrLog(@"error loading kernel function (scope_waveform_clear): %@", error);
        }
    } else {
        ErrLog(@"kernel function (scope_waveform_clear) not found");
    }
    
    
    id<MTLFunction> blendFunction = [defaultLibrary newFunctionWithName:@"scope_waveform_blend"];
    if (blendFunction) {
        NSError* error;
        kBlendComputePipelineState = [kDevice newComputePipelineStateWithFunction:blendFunction error:&error];
        if (error) {
            ErrLog(@"error loading kernel function (scope_waveform_blend): %@", error);
        }
    } else {
        ErrLog(@"kernel function (scope_waveform_blend) not found");
    }
    
    
    id<MTLFunction> paradeFunction = [defaultLibrary newFunctionWithName:@"scope_waveform_parade"];
    if (paradeFunction) {
        NSError* error;
        kParadeComputePipelineState = [kDevice newComputePipelineStateWithFunction:paradeFunction error:&error];
        if (error) {
            ErrLog(@"error loading kernel function (scope_waveform_parade): %@", error);
        }
    } else {
        ErrLog(@"kernel function (scope_waveform_parade) not found");
    }
    
    
    id<MTLFunction> lumaFunction = [defaultLibrary newFunctionWithName:@"scope_waveform_luma"];
    if (lumaFunction) {
        NSError* error;
        kLumaComputePipelineState = [kDevice newComputePipelineStateWithFunction:lumaFunction error:&error];
        if (error) {
            ErrLog(@"error loading kernel function (scope_waveform_luma): %@", error);
        }
    } else {
        ErrLog(@"kernel function (scope_waveform_luma) not found");
    }
    
}

+ (BOOL)processWithInputs:(NSArray<id<CIImageProcessorInput>> *)inputs arguments:(NSDictionary<NSString *,id> *)arguments output:(id<CIImageProcessorOutput>)output error:(NSError * _Nullable *)error
{
    id<MTLComputePipelineState> renderComputerState = kBlendComputePipelineState;
    
    NSNumber* type = arguments[@"type"];
    if (type) {
        switch (type.integerValue) {
            case kFTWaveformScopeTypeBlend:
            default:
                renderComputerState = kBlendComputePipelineState;
                break;
                
            case kFTWaveformScopeTypeParade:
                renderComputerState = kParadeComputePipelineState;
                break;
                
            case kFTWaveformScopeTypeLuminance:
                renderComputerState = kLumaComputePipelineState;
                break;
        }
    }
    
    if (!kWaveformComputePipelineState || !renderComputerState) {
        return NO;
    }
    
    id<CIImageProcessorInput> input = inputs.firstObject;
    
    id<MTLCommandBuffer> commandBuffer = output.metalCommandBuffer;
    commandBuffer.label = @"com.martinhering.WaveformKernel";
    id<MTLTexture> inputTexture = input.metalTexture;
    id<MTLTexture> outputTexture = output.metalTexture;
    
    
    MTLSize threadsPerGrid = MTLSizeMake(inputTexture.width, inputTexture.height, 1);
    
    NSUInteger w = kWaveformComputePipelineState.threadExecutionWidth;
    NSUInteger h = kWaveformComputePipelineState.maxTotalThreadsPerThreadgroup / w;
    MTLSize threadsPerThreadgroup = MTLSizeMake(w, h, 1);
    
    
    size_t columnBufSize = sizeof(UInt)*inputTexture.width*inputTexture.height;
    id<MTLBuffer> columnDataRed = [kDevice newBufferWithLength:columnBufSize options:0];
    id<MTLBuffer> columnDataGreen = [kDevice newBufferWithLength:columnBufSize options:0];
    id<MTLBuffer> columnDataBlue = [kDevice newBufferWithLength:columnBufSize options:0];
        
    id<MTLComputeCommandEncoder> computeEncoder;
    
    computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:kWaveformComputePipelineState];
    [computeEncoder setTexture:inputTexture atIndex:0];
    [computeEncoder setBuffer:columnDataRed offset:0 atIndex:0];
    [computeEncoder setBuffer:columnDataGreen offset:0 atIndex:1];
    [computeEncoder setBuffer:columnDataBlue offset:0 atIndex:2];
    [computeEncoder setSamplerState:kSamplerState atIndex:0];
    [computeEncoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [computeEncoder endEncoding];
    
    computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:renderComputerState];
    [computeEncoder setTexture:inputTexture atIndex:0];
    [computeEncoder setTexture:outputTexture atIndex:1];
    [computeEncoder setBuffer:columnDataRed offset:0 atIndex:0];
    [computeEncoder setBuffer:columnDataGreen offset:0 atIndex:1];
    [computeEncoder setBuffer:columnDataBlue offset:0 atIndex:2];
    [computeEncoder setSamplerState:kSamplerState atIndex:0];
    [computeEncoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [computeEncoder endEncoding];
    return YES;
}
@end
