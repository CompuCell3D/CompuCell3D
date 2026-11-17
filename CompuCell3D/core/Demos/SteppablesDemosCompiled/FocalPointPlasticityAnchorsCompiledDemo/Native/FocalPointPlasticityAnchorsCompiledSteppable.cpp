#include <cc3d/kernel.h>
#include <iostream>

namespace {
void step(CC3DKernelContext *ctx, void *) {
    if (!ctx) return;
    auto fpp = ctx->focalPointPlasticity.focalPointPlasticity();
    for (auto cell : ctx->cells) {
        if (cell.type != 1u) continue;
        if (ctx->mcs == 0u) {
            int id = fpp.createAnchor(cell, 8.0, 10.0, 100.0, 25.0f, 25.0f, 0.0f);
            std::cerr << "Created FPP anchor " << id << " for cell " << static_cast<long>(cell.id) << std::endl;
        }
        if (ctx->mcs == 50u) fpp.setAnchor(cell, 0, 12.0, 12.0, 100.0);
        if (ctx->mcs == 90u) fpp.eraseAnchor(cell, 0);
    }
}
}

extern "C" CC3D_STEPPABLE_EXPORT const CC3DSteppableV1 *cc3d_get_steppable_v1() {
    static const CC3DSteppableV1 api = {CC3D_KERNEL_ABI_VERSION, nullptr, nullptr, &step, nullptr, nullptr};
    return &api;
}
