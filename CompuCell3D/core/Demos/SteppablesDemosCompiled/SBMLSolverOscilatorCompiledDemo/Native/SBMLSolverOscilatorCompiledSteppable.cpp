#include <cc3d/kernel.h>

#include <iostream>

namespace {
constexpr unsigned char NONCONDENSING = 2;

void start(CC3DKernelContext *, void *) {}

void step(CC3DKernelContext *ctx, void *) {
    if (!ctx) {
        return;
    }

    bool printedSample = false;

    for (auto cell : ctx->cells) {
        if (static_cast<unsigned char>(cell.type) != NONCONDENSING) {
            continue;
        }

        auto model = ctx->sbml.find(cell, "OSCIL");
        if (!model) {
            continue;
        }

        double s1 = 0.0;
        if (!model.get("S1", &s1)) {
            continue;
        }

        cell.targetVolume = static_cast<float>(25.0 + 10.0 * s1);

        if (ctx->mcs == 3) {
            model.set("S1", 1.3);
            model.get("S1", &s1);
        }

        if (!printedSample && ctx->mcs < 10) {
            std::cerr << "C++ SBML steppable MCS " << ctx->mcs
                      << " cell " << static_cast<long>(cell.id)
                      << " OSCIL.S1=" << s1
                      << " targetVolume=" << static_cast<float>(cell.targetVolume)
                      << std::endl;
            printedSample = true;
        }

        model.timestep();
    }
}

void finish(CC3DKernelContext *, void *) {}
} // namespace

extern "C" CC3D_STEPPABLE_EXPORT const CC3DSteppableV1 *cc3d_get_steppable_v1() {
    static const CC3DSteppableV1 api = {
        CC3D_KERNEL_ABI_VERSION,
        nullptr,
        &start,
        &step,
        &finish,
        nullptr,
    };
    return &api;
}
