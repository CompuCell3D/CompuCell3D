#include <cc3d/kernel.h>

#include <cstdio>
#include <iostream>
#include <ostream>

namespace {

    using namespace std;
    constexpr unsigned char TUMOR = 1u;
    using namespace std;
    struct FieldGrowthState {
        uint64_t executionCount;
    };

    void *create() {
        return new FieldGrowthState{0u};
    }


    void start(CC3DKernelContext *ctx, void *statePtr) {
        using namespace std;

        auto seededTumor = ctx->newCell(TUMOR);
        seededTumor.targetVolume = 25.0f;
        seededTumor.lambdaVolume = 2.0f;
        ctx->cellField.assign(1, 6, 1, 6, 0, 1, seededTumor);

        for (auto cell : ctx->cells) {
            if (cell.type == TUMOR) {
                cell.targetVolume = 25.0f;
                cell.lambdaVolume = 2.0f;
            }
        }

        std::printf("Compiled field growth start\n");
    }

    void step(CC3DKernelContext *ctx, void *statePtr) {
        for (auto cell : ctx->cells) {

            cerr<<"C++ cell (id, type, targetVolume, lambdaVolume) = "<<cell.id<<" "<<cell.type<<" "<<cell.targetVolume<<" "<<cell.lambdaVolume<<endl;

        }
    }
    void finish(CC3DKernelContext *ctx, void *statePtr) {
        auto *state = static_cast<FieldGrowthState *>(statePtr);
        std::printf("Compiled field growth finish mcs=%llu execution=%llu\n",
                    static_cast<unsigned long long>(ctx->mcs),
                    static_cast<unsigned long long>(state->executionCount));
    }

    void destroy(void *statePtr) {

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
