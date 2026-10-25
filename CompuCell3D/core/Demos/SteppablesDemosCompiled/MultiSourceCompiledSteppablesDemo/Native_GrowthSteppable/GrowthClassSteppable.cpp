#include <cc3d/kernel.h>

#include <iostream>

#include "Utilities/Vector2D.h"

namespace {
    constexpr uint8_t TUMOR = 1u;

    class GrowthSteppable {
    public:
        void start(CC3DKernelContext *ctx) {
            executionCount_ = 0u;

            int count = 0;
            for (auto cell : ctx->cells) {
                cell.targetVolume = 25;
                cell.lambdaVolume = 10.0;
                // cell.targetSurface = 30.0;
                // cell.lambdaSurface = 10.0;

                if (count < 10) {
                    std::cerr << "growth cell.id=" << cell.id
                              << " targetVol=" << cell.targetVolume
                              << " targetSur=" << cell.targetSurface
                              << " lambdaSur=" << cell.lambdaSurface
                              << " surface=" << cell.surface << std::endl;
                }
                ++count;
            }

            std::cerr << "Multi-source growth start" << std::endl;
        }

        void step(CC3DKernelContext *ctx) {
            for (auto cell : ctx->cells) {
                if (cell.type == TUMOR) {
                    cell.targetVolume += 0.2f;

                    const Vector2D centerOfMass(cell.xCOM, cell.yCOM);
                    if (executionCount_ == 0u) {
                        std::cout << "growth vector utility cell.id=" << cell.id
                                  << " center_norm_squared=" << centerOfMass.normSquared()
                                  << std::endl;
                    }
                }
            }

            ++executionCount_;
            std::cout << "Multi-source growth step mcs="
                      << static_cast<unsigned long long>(ctx->mcs)
                      << " execution="
                      << static_cast<unsigned long long>(executionCount_)
                      << std::endl;
        }

        void finish(CC3DKernelContext *ctx) {
            std::cout << "Multi-source growth finish mcs="
                      << static_cast<unsigned long long>(ctx->mcs)
                      << " execution="
                      << static_cast<unsigned long long>(executionCount_)
                      << std::endl;
        }

    private:
        uint64_t executionCount_ = 0u;
    };

    void *create() { return new GrowthSteppable(); }

    void start(CC3DKernelContext *ctx, void *state) {
        static_cast<GrowthSteppable *>(state)->start(ctx);
    }

    void step(CC3DKernelContext *ctx, void *state) {
        static_cast<GrowthSteppable *>(state)->step(ctx);
    }

    void finish(CC3DKernelContext *ctx, void *state) {
        static_cast<GrowthSteppable *>(state)->finish(ctx);
    }

    void destroy(void *state) {
        delete static_cast<GrowthSteppable *>(state);
    }

    const CC3DSteppableV1 api = {
        CC3D_KERNEL_ABI_VERSION, &create, &start, &step, &finish, &destroy
    };
}

extern "C" CC3D_STEPPABLE_EXPORT const CC3DSteppableV1 *cc3d_get_steppable_v1() {
    return &api;
}
