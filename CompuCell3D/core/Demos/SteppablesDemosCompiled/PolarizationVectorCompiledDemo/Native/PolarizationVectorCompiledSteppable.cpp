#include <cc3d/kernel.h>
#include <iostream>

namespace {
void start(CC3DKernelContext *ctx, void *) {
    auto polarization = ctx->polarizationVector.polarizationVector();
    for (auto cell : ctx->cells) {
        if (cell.type == 1u) polarization.set(cell, 1.0f, 1.0f, 0.0f);
    }
    std::cerr << "PolarizationVector C++ steppable initialized" << std::endl;
}
void step(CC3DKernelContext *ctx, void *) {
    auto polarization = ctx->polarizationVector.polarizationVector();
    for (auto cell : ctx->cells) {
        if (cell.type == 1u) {
            auto vector = polarization.get(cell);
            std::cerr << "cell " << cell.id << " polarization = (" << vector[0] << ", " << vector[1] << ", " << vector[2] << ")" << std::endl;
            break;
        }
    }
}
const CC3DSteppableV1 api{CC3D_KERNEL_ABI_VERSION, nullptr, &start, &step, nullptr, nullptr};
}
extern "C" CC3D_STEPPABLE_EXPORT const CC3DSteppableV1 *cc3d_get_steppable_v1() { return &api; }
