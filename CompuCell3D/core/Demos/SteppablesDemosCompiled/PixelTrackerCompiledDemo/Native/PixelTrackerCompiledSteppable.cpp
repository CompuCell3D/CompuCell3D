#include <cc3d/kernel.h>

#include <iostream>

namespace {

void step(CC3DKernelContext *ctx, void *) {
    if (!ctx) {
        return;
    }

    bool printed = false;
    for (auto cell : ctx->cells) {
        if (static_cast<unsigned char>(cell.type) == 0) {
            continue;
        }

        unsigned int pixelCount = 0;
        cc3d::kernel::Pixel firstPixel;
        bool hasFirstPixel = false;

        for (auto pixel : ctx->pixels.forCell(cell)) {
            if (!hasFirstPixel) {
                firstPixel = pixel;
                hasFirstPixel = true;
            }
            ++pixelCount;
        }

        if (hasFirstPixel && !printed) {
            std::cerr << "C++ PixelTracker MCS " << ctx->mcs
                      << " cell " << static_cast<long>(cell.id)
                      << " pixels=" << pixelCount
                      << " first=(" << firstPixel.x << ", " << firstPixel.y << ", " << firstPixel.z << ")"
                      << std::endl;
            printed = true;
        }
    }
}

} // namespace

extern "C" CC3D_STEPPABLE_EXPORT const CC3DSteppableV1 *cc3d_get_steppable_v1() {
    static const CC3DSteppableV1 api = {
        CC3D_KERNEL_ABI_VERSION,
        nullptr,
        nullptr,
        &step,
        nullptr,
        nullptr,
    };
    return &api;
}
