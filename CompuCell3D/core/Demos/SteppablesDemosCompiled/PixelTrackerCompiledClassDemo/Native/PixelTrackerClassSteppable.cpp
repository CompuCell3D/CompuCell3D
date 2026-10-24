#include <cc3d/kernel.h>

#include <iostream>

namespace {

class PixelTrackerSteppable {
public:
    void start(CC3DKernelContext *) {}

    void step(CC3DKernelContext *ctx) {
        if (!ctx) return;

        bool printed = false;
        for (auto cell : ctx->cells) {
            if (static_cast<unsigned char>(cell.type) == 0) continue;

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
                          << " first=(" << firstPixel.x << ", "
                          << firstPixel.y << ", " << firstPixel.z << ")"
                          << std::endl;
                printed = true;
            }
        }
    }

    void finish(CC3DKernelContext *) {}
};

void *create() { return new PixelTrackerSteppable(); }

void start(CC3DKernelContext *ctx, void *state) {
    static_cast<PixelTrackerSteppable *>(state)->start(ctx);
}

void step(CC3DKernelContext *ctx, void *state) {
    static_cast<PixelTrackerSteppable *>(state)->step(ctx);
}

void finish(CC3DKernelContext *ctx, void *state) {
    static_cast<PixelTrackerSteppable *>(state)->finish(ctx);
}

void destroy(void *state) {
    delete static_cast<PixelTrackerSteppable *>(state);
}

const CC3DSteppableV1 api = {
    CC3D_KERNEL_ABI_VERSION,
    &create,
    &start,
    &step,
    &finish,
    &destroy,
};

} // namespace

extern "C" CC3D_STEPPABLE_EXPORT const CC3DSteppableV1 *cc3d_get_steppable_v1() {
    return &api;
}
