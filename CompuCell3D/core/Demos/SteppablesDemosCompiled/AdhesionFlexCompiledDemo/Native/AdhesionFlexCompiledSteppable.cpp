#include <cc3d/kernel.h>

#include <iostream>

namespace {

void printMolecules(const cc3d::kernel::AdhesionFlex &adhesion) {
    std::cerr << "C++ AdhesionFlex molecules:";
    for (size_t i = 0; i < adhesion.moleculeCount(); ++i) {
        const char *name = adhesion.moleculeName(i);
        if (name) {
            std::cerr << " " << name;
        }
    }
    std::cerr << std::endl;
}

void step(CC3DKernelContext *ctx, void *) {
    if (!ctx) {
        return;
    }

    const cc3d::kernel::AdhesionFlex adhesion(&ctx->adhesionFlex);
    if (ctx->mcs == 0u) {
        printMolecules(adhesion);
    }
    const float mediumDensities[] = {1.4f, 3.1f, 18.1f};
    adhesion.assignMediumVector(mediumDensities, 3u);
    adhesion.setMedium("NCam", 2.8f);
    adhesion.setMedium(2u, 16.8f);

    bool printed = false;
    for (auto cell : ctx->cells) {
        const auto type = static_cast<unsigned char>(cell.type);
        if (type == 1u) {
            const float densities[] = {3.4f, 2.1f, 12.1f};
            adhesion.assignVector(cell, densities, 3u);

            if (!printed) {
                std::cerr << "C++ AdhesionFlex MCS " << ctx->mcs
                          << " cell " << static_cast<long>(cell.id)
                          << " NCad=" << adhesion.get(cell, "NCad")
                          << " Int=" << adhesion.get(cell, 2u)
                          << " medium Int=" << adhesion.medium("Int")
                          << std::endl;
                printed = true;
            }
        } else if (type == 2u) {
            adhesion.set(cell, "NCad", 11.2f);
            adhesion.set(cell, 2u, 11.1f);
        }
    }
}

} // namespace

extern "C" CC3D_STEPPABLE_EXPORT const CC3DSteppableV1 *cc3d_get_steppable_v1() {
    static const CC3DSteppableV1 api = {
        CC3D_KERNEL_ABI_VERSION,
        nullptr,
        nullptr,
        &step,
        nullptr,
        nullptr,
    };
    return &api;
}
