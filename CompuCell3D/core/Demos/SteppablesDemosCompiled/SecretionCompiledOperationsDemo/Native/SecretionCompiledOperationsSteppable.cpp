#include <cc3d/kernel.h>
#include <iostream>

namespace {

void step(CC3DKernelContext *ctx, void *) {
    auto attr = ctx->secretion.fieldSecretor("ATTR");
    const std::vector<unsigned char> wall{3};
    const std::vector<unsigned char> medium{0};

    for (auto cell : ctx->cells) {
        if (cell.type != 3u) continue; // Wall

        // These calls mirror bacterium_macrophage_2D_secretion_steppables.py.
        const bool insideContact = attr.secreteInsideCellAtBoundaryOnContactWith(cell, 300.0f, wall);
        const bool outsideContact = attr.secreteOutsideCellAtBoundaryOnContactWith(cell, 300.0f, medium);
        const bool inside = attr.secreteInsideCell(cell, 300.0f);
        const bool insideBoundary = attr.secreteInsideCellAtBoundary(cell, 300.0f);
        const bool outsideBoundary = attr.secreteOutsideCellAtBoundary(cell, 500.0f);
        const bool com = attr.secreteInsideCellAtCOM(cell, 300.0f);

        if (ctx->mcs % 10 == 0) {
            std::cerr << "MCS " << ctx->mcs << " wall=" << cell.id
                      << " inside_contact=" << insideContact
                      << " outside_contact=" << outsideContact
                      << " inside=" << inside
                      << " inside_boundary=" << insideBoundary
                      << " outside_boundary=" << outsideBoundary
                      << " com=" << com
                      << " field_integral=" << attr.totalFieldIntegral()
                      << std::endl;
        }
    }
}

const CC3DSteppableV1 api{
    CC3D_KERNEL_ABI_VERSION, nullptr, nullptr, &step, nullptr, nullptr
};

}

extern "C" CC3D_STEPPABLE_EXPORT const CC3DSteppableV1 *cc3d_get_steppable_v1() {
    return &api;
}
