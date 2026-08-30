#include <cc3d/kernel.h>

#include <cstdio>
#include <iostream>
#include <ostream>

namespace {
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
        auto *state = static_cast<FieldGrowthState *>(statePtr);
        state->executionCount = 0u;

        for (auto cell : ctx->cells) {
            if (cell.type == TUMOR) {
                cell.targetVolume = 25.0f;
                cell.lambdaVolume = 2.0f;
            }
        }

        std::printf("Compiled field growth start\n");
    }

    void step(CC3DKernelContext *ctx, void *statePtr) {
        auto *state = static_cast<FieldGrowthState *>(statePtr);
        auto fgf = ctx->scalarFields["FGF"];
        auto dim = ctx->cellField.dim();

        if (!fgf) {
            std::printf("Compiled field growth missing field FGF\n");
            return;
        }

        bool printedSample = false;
        unsigned long medium_pixels = 0;
        unsigned long cell_pixels = 0;
        for (int z = 0; z < dim.z; ++z) {
            for (int y = 0; y < dim.y; ++y) {
                for (int x = 0; x < dim.x; ++x) {
                    auto pixelCell = ctx->cellField(x, y, z);
                    if (!pixelCell) {
                        ++medium_pixels;
                        continue;
                    }
                    ++cell_pixels;
                    // auto cell = pixelCell.value();
                    // if (!printedSample && cell.type == TUMOR) {
                    //     std::printf("Compiled field growth pixel sample x=%d y=%d z=%d targetVolume=%g\n",
                    //                 x, y, z, static_cast<double>(cell.targetVolume));
                    //     printedSample = true;
                    // }
                }
            }
        }
        cerr<<"medium_pixels="<<medium_pixels<<" cell_pixels="<<cell_pixels<<" totalPixels="<<cell_pixels+medium_pixels<<endl;
        // for (auto cell : ctx->cells) {
        //     if (cell.type != TUMOR) {
        //         continue;
        //     }
        //
        //     int x = static_cast<int>(cell.xCOM);
        //     int y = static_cast<int>(cell.yCOM);
        //     int z = static_cast<int>(cell.zCOM);
        //
        //     float concentration = fgf(x, y, z);
        //     if (concentration > 0.0f) {
        //         cell.targetVolume += 0.05f * concentration / (10.0f + concentration);
        //     }
        // }
        //
        // ++state->executionCount;
        // std::printf("Compiled field growth step mcs=%llu execution=%llu\n",
        //             static_cast<unsigned long long>(ctx->mcs),
        //             static_cast<unsigned long long>(state->executionCount));
    }

    void finish(CC3DKernelContext *ctx, void *statePtr) {
        auto *state = static_cast<FieldGrowthState *>(statePtr);
        std::printf("Compiled field growth finish mcs=%llu execution=%llu\n",
                    static_cast<unsigned long long>(ctx->mcs),
                    static_cast<unsigned long long>(state->executionCount));
    }

    void destroy(void *statePtr) {
        delete static_cast<FieldGrowthState *>(statePtr);
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
