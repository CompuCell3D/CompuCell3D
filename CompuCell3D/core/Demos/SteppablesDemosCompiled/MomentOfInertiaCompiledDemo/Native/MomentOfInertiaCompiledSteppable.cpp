#include <cc3d/kernel.h>

#include <iostream>

namespace {
    void start(CC3DKernelContext *, void *) {
        std::cerr << "Compiled MomentOfInertia start\n";
    }

    void step(CC3DKernelContext *ctx, void *) {
        auto momentOfInertia = ctx->momentOfInertia.momentOfInertia();
        for (auto cell : ctx->cells) {
            const auto axes = momentOfInertia.getSemiaxes(cell);
            const auto axesXY = momentOfInertia.getSemiaxesXY(cell);
            const auto axesXZ = momentOfInertia.getSemiaxesXZ(cell);
            const auto axesYZ = momentOfInertia.getSemiaxesYZ(cell);
            const auto axes3D = momentOfInertia.getSemiaxes3D(cell);

            std::cerr << "MomentOfInertia cell=" << static_cast<long>(cell.id)
                      << " semiaxes=(" << axes[0] << ',' << axes[1] << ',' << axes[2]
                      << ") XY=(" << axesXY[0] << ',' << axesXY[1] << ',' << axesXY[2]
                      << ") XZ=(" << axesXZ[0] << ',' << axesXZ[1] << ',' << axesXZ[2]
                      << ") YZ=(" << axesYZ[0] << ',' << axesYZ[1] << ',' << axesYZ[2]
                      << ") 3D=(" << axes3D[0] << ',' << axes3D[1] << ',' << axes3D[2] << ")\n";
        }
    }

    const CC3DSteppableV1 api = {
        CC3D_KERNEL_ABI_VERSION, nullptr, &start, &step, nullptr, nullptr
    };
}

extern "C" CC3D_STEPPABLE_EXPORT const CC3DSteppableV1 *cc3d_get_steppable_v1() {
    return &api;
}
