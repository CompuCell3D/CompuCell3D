#include <cc3d/kernel.h>

#include <cstdio>
#include <iostream>
#include <ostream>

namespace {
    constexpr unsigned char TUMOR = 1u;

    class FieldDrivenGrowthSteppable {
    public:
        void start(CC3DKernelContext *ctx) {
            executionCount_ = 0u;

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

        void step(CC3DKernelContext *ctx) {
            auto fgf = ctx->scalarFields["FGF"];
            auto dim = ctx->cellField.dim();

            if (!fgf) {
                std::printf("Compiled field growth missing field FGF\n");
                return;
            }

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
                    }
                }
            }

            std::cerr << "medium_pixels=" << medium_pixels
                      << " cell_pixels=" << cell_pixels
                      << " totalPixels=" << cell_pixels + medium_pixels
                      << std::endl;
        }

        void finish(CC3DKernelContext *ctx) {
            std::printf("Compiled field growth finish mcs=%llu execution=%llu\n",
                        static_cast<unsigned long long>(ctx->mcs),
                        static_cast<unsigned long long>(executionCount_));
        }

    private:
        uint64_t executionCount_ = 0u;
    };

    void *create() {
        return new FieldDrivenGrowthSteppable();
    }

    void start(CC3DKernelContext *ctx, void *state) {
        static_cast<FieldDrivenGrowthSteppable *>(state)->start(ctx);
    }

    void step(CC3DKernelContext *ctx, void *state) {
        static_cast<FieldDrivenGrowthSteppable *>(state)->step(ctx);
    }

    void finish(CC3DKernelContext *ctx, void *state) {
        static_cast<FieldDrivenGrowthSteppable *>(state)->finish(ctx);
    }

    void destroy(void *state) {
        delete static_cast<FieldDrivenGrowthSteppable *>(state);
    }

    const CC3DSteppableV1 api = {
        CC3D_KERNEL_ABI_VERSION, &create, &start, &step, &finish, &destroy
    };
}

extern "C" CC3D_STEPPABLE_EXPORT const CC3DSteppableV1 *cc3d_get_steppable_v1() {
    return &api;
}
