#include <cc3d/kernel.h>
#include <iostream>

namespace {
void step(CC3DKernelContext *ctx, void *) {
    if (!ctx) return;
    auto cells = ctx->cells;
    auto fpp = ctx->focalPointPlasticity.focalPointPlasticity();
    auto first = cells.begin();
    if (!(first != cc3d::kernel::CellSentinel{})) return;
    auto a = *first;
    ++first;
    if (!(first != cc3d::kernel::CellSentinel{})) return;
    auto b = *first;
    if (ctx->mcs == 0u) {
        fpp.create(a, b, 10.0, 15.0, 40.0);
        std::cerr << "Created FPP link: lambda=" << fpp.lambda(a, b)
                  << " target=" << fpp.target(a, b) << std::endl;
    }
    if (ctx->mcs == 50u) fpp.set(a, b, 12.0, 18.0, 40.0);
    if (ctx->mcs == 90u) fpp.erase(a, b);
}
}

extern "C" CC3D_STEPPABLE_EXPORT const CC3DSteppableV1 *cc3d_get_steppable_v1() {
    static const CC3DSteppableV1 api = {CC3D_KERNEL_ABI_VERSION, nullptr, nullptr, &step, nullptr, nullptr};
    return &api;
}
