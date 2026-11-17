#include <cc3d/kernel.h>
#include <iostream>

namespace {
void start(CC3DKernelContext *ctx, void *) {
    auto chemotaxis = ctx->chemotaxis.chemotaxis();
    for (auto cell : ctx->cells) {
        if (cell.type == 2u) {
            auto data = chemotaxis.add(cell, "ATTR");
            data.setLambda(30.0f);
            data.assignChemotactTowardsTypes({0, 1});
            std::cerr << "Chemotaxis C++ data initialized for cell " << cell.id
                      << ": lambda=" << data.lambda()
                      << ", saturationCoef=" << data.saturationCoef()
                      << ", formula=" << (data.formulaName() ? data.formulaName() : "<none>")
                      << std::endl;
            break;
        }
    }
}
void step(CC3DKernelContext *ctx, void *) {
    auto chemotaxis = ctx->chemotaxis.chemotaxis();
    for (auto cell : ctx->cells) {
        if (cell.type == 2u) {
            auto data = chemotaxis.get(cell, "ATTR");
            if (data) {
                data.setLambda(data.lambda() - 0.1f);
                if (ctx->mcs % 10 == 0) {
                    auto types = data.getChemotactTowardsVectorTypes();
                    std::cerr << "MCS " << ctx->mcs << " chemotaxis cell " << cell.id
                              << ": lambda=" << data.lambda()
                              << ", saturationCoef=" << data.saturationCoef()
                              << ", formula=" << (data.formulaName() ? data.formulaName() : "<none>")
                              << ", towardTypes=[";
                    for (size_t i = 0; i < types.size(); ++i) {
                        if (i) std::cerr << ", ";
                        std::cerr << types[i];
                    }
                    std::cerr << "]" << std::endl;
                }
            }
            break;
        }
    }
}
const CC3DSteppableV1 api{CC3D_KERNEL_ABI_VERSION, nullptr, &start, &step, nullptr, nullptr};
}
extern "C" CC3D_STEPPABLE_EXPORT const CC3DSteppableV1 *cc3d_get_steppable_v1() { return &api; }
