#include <cc3d/kernel.h>

#include <iostream>

namespace {

void step(CC3DKernelContext *ctx, void *) {
    if (!ctx) return;

    const auto length = ctx->lengthConstraint.lengthConstraint();
    for (auto cell : ctx->cells) {
        if (cell.type == 1u) {
            if (ctx->mcs == 0u) {
                std::cerr << "C++ LengthConstraint cell " << static_cast<long>(cell.id)
                          << " lambda=" << length.lambda(cell)
                          << " target=" << length.target(cell)
                          << " minorTarget=" << length.minorTarget(cell) << std::endl;
            }
            length.set(cell, 20.0, 20.0, 0.0);
        } else if (cell.type == 2u) {
            length.set(cell, 20.0, 30.0, 0.0);
        }
    }
}

} // namespace

extern "C" CC3D_STEPPABLE_EXPORT const CC3DSteppableV1 *cc3d_get_steppable_v1() {
    static const CC3DSteppableV1 api = {
        CC3D_KERNEL_ABI_VERSION, nullptr, nullptr, &step, nullptr, nullptr
    };
    return &api;
}
