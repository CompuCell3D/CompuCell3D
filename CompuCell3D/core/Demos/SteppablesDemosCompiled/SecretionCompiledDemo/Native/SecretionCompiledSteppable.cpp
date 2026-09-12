#include <cc3d/kernel.h>
#include <iostream>

using namespace std;

namespace {
void start(CC3DKernelContext *ctx, void *) {
    cerr<<"start fcn C++ steppable"<<endl;
    auto secretor = ctx->secretion.fieldSecretor("ATTR");
    // Cell initializers run after the compiled steppable's start callback, so
    // there may be no cells here yet. Secretion is therefore performed in step().
}
void step(CC3DKernelContext *ctx, void *) {
    cerr<<"step fcn C++ steppable"<<endl;
    auto secretor = ctx->secretion.fieldSecretor("ATTR");
    for (auto cell : ctx->cells) {
        if (cell.type == 1u) {
            auto inside = secretor.secreteInsideCellTotalCount(cell, 1.0f);
            auto boundary = secretor.secreteInsideCellAtBoundaryOnContactWithTotalCount(cell, 5.0f, {0});
            std::cerr << "MCS " << ctx->mcs << ": inside amount=" << inside.totalAmount
                      << ", boundary amount=" << boundary.totalAmount
                      << ", field integral=" << secretor.totalFieldIntegral() << std::endl;
            break;
        }
    }
}
const CC3DSteppableV1 api{CC3D_KERNEL_ABI_VERSION, nullptr, &start, &step, nullptr, nullptr};
}
extern "C" CC3D_STEPPABLE_EXPORT const CC3DSteppableV1 *cc3d_get_steppable_v1() { return &api; }
