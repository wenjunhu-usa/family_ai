#import <AppKit/AppKit.h>
#import <CoreGraphics/CoreGraphics.h>
#import <CoreImage/CoreImage.h>
#import <ImageIO/ImageIO.h>
#import <Vision/Vision.h>

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc >= 3 && strcmp(argv[1], "detect") == 0) {
            NSURL *imageURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[2]]];
            CGImageSourceRef source = CGImageSourceCreateWithURL((__bridge CFURLRef)imageURL, NULL);
            CGImageRef sourceImage = source ? CGImageSourceCreateImageAtIndex(source, 0, NULL) : NULL;
            BOOL nearlyBlack = NO;
            if (sourceImage) {
                CIImage *ciImage = [[CIImage alloc] initWithCGImage:sourceImage];
                CIFilter *average = [CIFilter filterWithName:@"CIAreaAverage"];
                [average setValue:ciImage forKey:kCIInputImageKey];
                [average setValue:[CIVector vectorWithCGRect:ciImage.extent] forKey:kCIInputExtentKey];
                uint8_t pixel[4] = {0};
                CIContext *context = [CIContext contextWithOptions:nil];
                CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
                [context render:average.outputImage toBitmap:pixel rowBytes:4 bounds:CGRectMake(0, 0, 1, 1)
                         format:kCIFormatRGBA8 colorSpace:colorSpace];
                CGColorSpaceRelease(colorSpace);
                nearlyBlack = ((pixel[0] + pixel[1] + pixel[2]) / 3.0) < 8.0;
                CGImageRelease(sourceImage);
            }
            if (source) CFRelease(source);
            VNDetectHumanRectanglesRequest *request = [VNDetectHumanRectanglesRequest new];
            request.upperBodyOnly = NO;
            VNRecognizeTextRequest *textRequest = [VNRecognizeTextRequest new];
            textRequest.recognitionLevel = VNRequestTextRecognitionLevelFast;
            VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithURL:imageURL options:@{}];
            NSError *error = nil;
            if (![handler performRequests:@[request, textRequest] error:&error]) {
                fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
                return 5;
            }
            NSArray<VNHumanObservation *> *results = request.results ?: @[];
            float confidence = 0;
            for (VNHumanObservation *observation in results) confidence = MAX(confidence, observation.confidence);
            NSMutableArray<NSString *> *recognized = [NSMutableArray array];
            for (VNRecognizedTextObservation *observation in textRequest.results ?: @[]) {
                VNRecognizedText *candidate = [[observation topCandidates:1] firstObject];
                if (candidate.string.length) [recognized addObject:candidate.string];
            }
            NSDictionary *payload = @{ @"person_count": @(results.count), @"confidence": @(confidence),
                                       @"text": [recognized componentsJoinedByString:@" "],
                                       @"nearly_black": @(nearlyBlack) };
            NSData *json = [NSJSONSerialization dataWithJSONObject:payload options:0 error:nil];
            fwrite(json.bytes, 1, json.length, stdout);
            printf("\n");
            return 0;
        }
        if (argc < 3 || (strcmp(argv[1], "capture") != 0 && strcmp(argv[1], "id") != 0)) {
            fprintf(stderr, "usage: camera-window-capture <id|capture> <provider> [output.png] | detect <image>\n");
            return 64;
        }
        NSString *target = [NSString stringWithUTF8String:argv[2]];
        NSString *output = argc >= 4 ? [NSString stringWithUTF8String:argv[3]] : nil;
        NSArray<NSString *> *owners = @[@"*"];
        CFArrayRef infoRef = CGWindowListCopyWindowInfo(
            kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements,
            kCGNullWindowID
        );
        NSArray *windows = CFBridgingRelease(infoRef);
        CGWindowID selected = 0;
        for (NSDictionary *window in windows) {
            NSString *owner = window[(id)kCGWindowOwnerName];
            NSNumber *layer = window[(id)kCGWindowLayer];
            if ([owners containsObject:owner] && layer.intValue == 0) {
                selected = [window[(id)kCGWindowNumber] unsignedIntValue];
                break;
            }
        }
        if (!selected) {
            fprintf(stderr, "allowed camera window is not visible\n");
            return 2;
        }
        if (strcmp(argv[1], "id") == 0) {
            printf("%u\n", selected);
            return 0;
        }
        CGImageRef image = CGWindowListCreateImage(
            CGRectNull, kCGWindowListOptionIncludingWindow, selected,
            kCGWindowImageBoundsIgnoreFraming | kCGWindowImageBestResolution
        );
        if (!image) {
            fprintf(stderr, "screen recording is not authorized\n");
            return 3;
        }
        CGImageRef outputImage = image;
        NSURL *url = [NSURL fileURLWithPath:output];
        CGImageDestinationRef destination = CGImageDestinationCreateWithURL(
            (__bridge CFURLRef)url, CFSTR("public.png"), 1, NULL
        );
        CGImageDestinationAddImage(destination, outputImage, NULL);
        BOOL ok = CGImageDestinationFinalize(destination);
        CFRelease(destination);
        CGImageRelease(image);
        if (!ok) return 4;
        printf("%s\n", output.UTF8String);
        return 0;
    }
}
