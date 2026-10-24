#include <cc3d/kernel.h>
#include <iostream>

#include <cstdio>

namespace {
    constexpr uint8_t TUMOR = 1u;

    class GrowthSteppable {
    public:
        void start(CC3DKernelContext *ctx) {
            executionCount_ = 0u;

            int count = 0;
            for (auto cell : ctx->cells) {
                cell.targetVolume = 25;
                cell.lambdaVolume = 2.0;
                cell.targetSurface = 30.0;
                cell.lambdaSurface = 1.0;

                if (count < 10) {
                    std::cerr << "demo" << std::endl;
                    std::cerr << "cell.id=" << cell.id
                              << " targetVol=" << cell.targetVolume
                              << " targetSur=" << cell.targetSurface
                              << " lambdaSur=" << cell.lambdaSurface
                              << " surface=" << cell.surface
                              << '\n';
                }
                ++count;
            }

            std::printf("Compiled growth start\n");
        }

        void step(CC3DKernelContext *ctx) {
            for (auto cell : ctx->cells) {
                if (cell.type == TUMOR) {
                    cell.targetVolume += 0.2f;
                    std::cerr << " cell.id=" << cell.id
                              << " xcom=" << cell.xCOM
                              << " yCOM=" << cell.yCOM << std::endl;
                }
            }

            ++executionCount_;
            std::printf("Compiled growth step mcs=%llu execution=%llu\n",
                        static_cast<unsigned long long>(ctx->mcs),
                        static_cast<unsigned long long>(executionCount_));
        }

        void finish(CC3DKernelContext *ctx) {
            std::printf("Compiled growth finish mcs=%llu execution=%llu\n",
                        static_cast<unsigned long long>(ctx->mcs),
                        static_cast<unsigned long long>(executionCount_));
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
