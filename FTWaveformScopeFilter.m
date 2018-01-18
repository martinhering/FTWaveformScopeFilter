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

#import "FTWaveformScopeFilter.h"
#import "FTWaveformScopeKernel.h"

NSString * const kFTWaveformScopeType = @"type";

@implementation FTWaveformScopeFilter {
    CIImage* inputImage;
    CIFilter* _resizeFilter;
    NSNumber* type;
}

- (id)init
{
    if (self = [super init]) {
        _resizeFilter = [CIFilter filterWithName:@"CILanczosScaleTransform"];
    }
    
    return self;
}

- (CIImage *)outputImage
{
    NSParameterAssert(inputImage != nil && [inputImage isKindOfClass:[CIImage class]]);
    
    if (!type) {
        type = @(0);
    }

    CGRect imageExtent = inputImage.extent;

    [_resizeFilter setValue:inputImage forKey:kCIInputImageKey];
    [_resizeFilter setValue:@(512/CGRectGetHeight(imageExtent)) forKey:kCIInputScaleKey];
    CIImage* resizedImage = [_resizeFilter valueForKey:kCIOutputImageKey];

    NSMutableDictionary* options = [[NSMutableDictionary alloc] init];

    CGRect resizeImageExtent = resizedImage.extent;
    options[kCIApplyOptionExtent] = @[@(0),@(0),@(CGRectGetWidth(resizeImageExtent)),@(CGRectGetHeight(resizeImageExtent))];

    NSError* error;
    CIImage *outputImage = [FTWaveformScopeKernel applyWithExtent:resizeImageExtent
                                                           inputs:@[resizedImage]
                                                        arguments:@{ kFTWaveformScopeType : type}
                                                            error:&error];

    return outputImage;
}

- (NSDictionary *)customAttributes
{
    return @{
             @"type" : @{kCIAttributeDefault : @(0), kCIAttributeType : kCIAttributeTypeScalar},
             };
}

@end
