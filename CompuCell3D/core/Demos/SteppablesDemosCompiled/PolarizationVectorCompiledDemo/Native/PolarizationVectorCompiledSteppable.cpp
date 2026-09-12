#include <cc3d/kernel.h>
#include <iostream>

namespace {
void start(CC3DKernelContext *ctx, void *) {
    auto polarization = ctx->polarizationVector.polarizationVector();
    auto orientation = ctx->cellOrientation.cellOrientation();
    for (auto cell : ctx->cells) {
        if (cell.type == 1u) {
            polarization.set(cell, 1.0f, 1.1f, 0.0f);
            orientation.setLambda(cell, 5.0);
            cell.targetVolume=25.;
            cell.lambdaVolume = 2.0;
        }
    }
    std::cerr << "PolarizationVector C++ steppable initialized" << std::endl;
}
void step(CC3DKernelContext *ctx, void *) {
    auto polarization = ctx->polarizationVector.polarizationVector();
    auto orientation = ctx->cellOrientation.cellOrientation();
    for (auto cell : ctx->cells) {
        if (cell.type == 1u) {
            if (ctx->mcs == 500) {
                orientation.setLambda(cell, 25.0);
                polarization.set(cell, 1.0f, -1.1f, 0.0f);
            }

            auto vector = polarization.get(cell);
            std::cerr << "cell " << cell.id << " polarization = (" << vector[0] << ", " << vector[1] << ", " << vector[2] << ")" << std::endl;
            std::cerr << "cell " << cell.id << " lambdaCellOrientation = " << orientation.getLambda(cell) << std::endl;
            break;
        }
    }
}
const CC3DSteppableV1 api{CC3D_KERNEL_ABI_VERSION, nullptr, &start, &step, nullptr, nullptr};
}
extern "C" CC3D_STEPPABLE_EXPORT const CC3DSteppableV1 *cc3d_get_steppable_v1() { return &api; }
