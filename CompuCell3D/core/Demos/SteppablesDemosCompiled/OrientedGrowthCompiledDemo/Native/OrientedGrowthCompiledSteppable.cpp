#include <cc3d/kernel.h>

#include <iostream>

namespace {
    struct OrientedGrowthState {
        unsigned long long calls = 0;
        bool configured = false;
    };

    void *create() {
        return new OrientedGrowthState;
    }

    void start(CC3DKernelContext *ctx, void *) {
        std::cerr << "Compiled OrientedGrowth start mcs=" << ctx->mcs << '\n';
    }

    void step(CC3DKernelContext *ctx, void *statePtr) {
        auto *state = static_cast<OrientedGrowthState *>(statePtr);
        auto orientedGrowth = ctx->orientedGrowth.orientedGrowth();

        if (!state->configured) {
            constexpr float axisX = 0.5f;
            constexpr float axisY = 0.8660254038f;

            for (auto cell : ctx->cells) {
                cell.lambdaVolume = 2.0f;
                cell.targetVolume = static_cast<float>(cell.volume);
                orientedGrowth.setElongationAxis(cell, axisX, axisY);
                orientedGrowth.setConstraintWidth(cell, 2.0f);
                orientedGrowth.setElongationEnabled(cell, true);

                std::cerr << "OrientedGrowth cell=" << static_cast<long>(cell.id)
                          << " axis=(" << orientedGrowth.getElongationAxis_X(cell)
                          << ',' << orientedGrowth.getElongationAxis_Y(cell)
                          << ") enabled=" << (orientedGrowth.getElongationEnabled(cell) ? 1 : 0) << '\n';
            }
            state->configured = true;
        }

        for (auto cell : ctx->cells) cell.targetVolume += 1.0f;
        ++state->calls;
        if (state->calls <= 3) {
            std::cerr << "Compiled OrientedGrowth step mcs=" << ctx->mcs << '\n';
        }
    }

    void finish(CC3DKernelContext *ctx, void *statePtr) {
        const auto *state = static_cast<const OrientedGrowthState *>(statePtr);
        std::cerr << "Compiled OrientedGrowth finish mcs=" << ctx->mcs
                  << " calls=" << state->calls << '\n';
    }

    void destroy(void *statePtr) {
        delete static_cast<OrientedGrowthState *>(statePtr);
    }

    const CC3DSteppableV1 api = {
        CC3D_KERNEL_ABI_VERSION, &create, &start, &step, &finish, &destroy
    };
}

extern "C" CC3D_STEPPABLE_EXPORT const CC3DSteppableV1 *cc3d_get_steppable_v1() {
    return &api;
}
