#include <cc3d/kernel.h>
#include <iostream>

#include <cstdio>

namespace {
    constexpr uint8_t TUMOR = 1u;

    struct GrowthState {
        uint64_t executionCount;
    };

    void *create() {
        return new GrowthState{0u};
    }

void start(CC3DKernelContext *ctx, void *statePtr) {
    auto *state = static_cast<GrowthState *>(statePtr);
    state->executionCount = 0u;

    int count = 0;
    for (auto cell : ctx->cells) {
        cell.targetVolume = 25;
        cell.lambdaVolume = 2.0;

        cell.targetSurface =30.;
        cell.lambdaSurface = 1.0;

        if (count < 10) {
            std::cerr<<"demo"<<std::endl;
            std::cerr << "cell.id=" << cell.id
                      << " targetVol=" << cell.targetVolume
                      << " targetSur=" << cell.targetSurface
                      << " lambdaSur=" << cell.lambdaSurface
                      <<" surface="<< cell.surface
                      << '\n';
        }
        count++;
    }

    std::printf("Compiled growth start\n");
}

//     void start(CC3DKernelContext *ctx, void *statePtr) {
//         auto *state = static_cast<GrowthState *>(statePtr);
//         state->executionCount = 0u;
//         int count = 0;
//         for (auto cell : ctx->cells){
//             if (count) > 10{
//                 break;
//                 }
//             std::cerr<<"cell.id="cell->id<<" targetVol="<<cell->targetVolume<<std::endl;
//             count++;
//             }
//         std::printf("Compiled growth start\n");
//     }

    void step(CC3DKernelContext *ctx, void *statePtr) {
        using namespace std;
        auto *state = static_cast<GrowthState *>(statePtr);

        for (auto cell : ctx->cells) {
            if (cell.type == TUMOR) {
                cell.targetVolume += 0.2f;
                cerr<<" cell.id="<<cell.id<<" xcom="<<cell.xCOM<<" yCOM="<<cell.yCOM<<endl;
            }
        }

        ++state->executionCount;
        std::printf("Compiled growth step mcs=%llu execution=%llu\n",
                    static_cast<unsigned long long>(ctx->mcs),
                    static_cast<unsigned long long>(state->executionCount));
    }

    void finish(CC3DKernelContext *ctx, void *statePtr) {
        auto *state = static_cast<GrowthState *>(statePtr);
        std::printf("Compiled growth finish mcs=%llu execution=%llu\n",
                    static_cast<unsigned long long>(ctx->mcs),
                    static_cast<unsigned long long>(state->executionCount));
    }

    void destroy(void *statePtr) {
        delete static_cast<GrowthState *>(statePtr);
    }

    const CC3DSteppableV1 api = {
        CC3D_KERNEL_ABI_VERSION,
        &create,
        &start,
        &step,
        &finish,
        &destroy
    };
}

extern "C" const CC3DSteppableV1 *cc3d_get_steppable_v1() {
    return &api;
}
