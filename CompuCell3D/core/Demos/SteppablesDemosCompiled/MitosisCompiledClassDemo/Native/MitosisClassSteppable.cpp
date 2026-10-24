// Use the repository ABI header directly. The Python environment may contain
// an older copy where Cell::type is read-only, although the C ABI supports
// changing a cell type after mitosis.
#include <cc3d/kernel.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

namespace {
    constexpr unsigned char CONDENSING = 1u;
    constexpr unsigned char CONDENSING_DIFFERENTIATED = 3u;
    constexpr float INITIAL_TARGET_VOLUME = 25.0f;
    constexpr float LAMBDA_VOLUME = 2.0f;

    class MitosisSteppable {
    public:
        void start(CC3DKernelContext *ctx) {
            initialize_cells(ctx);
        }

        void step(CC3DKernelContext *ctx) {
            if (!initialized_) initialize_cells(ctx);

            auto fgf = ctx->scalarFields["FGF"];
            if (!fgf) {
                if (!missingFgfReported_) {
                    std::cerr << "Compiled mitosis growth: FGF field is unavailable\n";
                    missingFgfReported_ = true;
                }
                return;
            }

            const auto dimensions = fgf.dim();
            (void)dimensions;
            std::vector<cc3d::kernel::Cell> cellsToDivide;
            for (auto cell : ctx->cells) {
                if (cell.type != CONDENSING) continue;

                int x = static_cast<int>(std::lround(static_cast<double>(cell.xCOM)));
                int y = static_cast<int>(std::lround(static_cast<double>(cell.yCOM)));
                int z = static_cast<int>(std::lround(static_cast<double>(cell.zCOM)));

                const float concentration = static_cast<float>(fgf(x, y, z));
                cell.targetVolume += 0.1f * concentration;

                if (cell.volume > 2 * INITIAL_TARGET_VOLUME) {
                    cellsToDivide.push_back(cell);
                }
            }

            auto mitosis = ctx->mitosis.mitosis();
            mitosis.setParentChildPositionFlag(1);
            std::bernoulli_distribution differentiate(0.5);
            for (const auto &cell : cellsToDivide) {
                const long parentId = cell.id;
                if (!mitosis.divideRandom(cell)) continue;

                auto parent = mitosis.parentCell();
                auto child = mitosis.childCell();
                parent.targetVolume = static_cast<float>(parent.targetVolume) / 2.0f;
                mitosis.cloneAttributes(parent, child);
                if (differentiate(random_)) {
                    child.type = CONDENSING_DIFFERENTIATED;
                }
                ++divisions_;
                std::cerr << "mitosis mcs=" << ctx->mcs
                          << " source=" << parentId
                          << " parent=" << static_cast<long>(parent.id)
                          << " child=" << static_cast<long>(child.id)
                          << " childType=" << static_cast<unsigned int>(child.type)
                          << '\n';
            }
        }

        void finish(CC3DKernelContext *) {
            std::cerr << "Compiled single-cell mitosis divisions="
                      << divisions_ << '\n';
        }

    private:
        void initialize_cells(CC3DKernelContext *ctx) {
            unsigned int cells = 0;
            for (auto cell : ctx->cells) {
                cell.targetVolume = INITIAL_TARGET_VOLUME;
                cell.lambdaVolume = LAMBDA_VOLUME;
                ++cells;
            }
            if (cells == 0) return;

            initialized_ = true;
            std::cerr << "Compiled mitosis growth start cells=" << cells
                      << " targetVolume=" << INITIAL_TARGET_VOLUME
                      << " lambdaVolume=" << LAMBDA_VOLUME << '\n';
        }

        unsigned long long divisions_ = 0;
        std::mt19937 random_{0xCC3D2026u};
        bool missingFgfReported_ = false;
        bool initialized_ = false;
    };

    void *create() { return new MitosisSteppable(); }

    void start(CC3DKernelContext *ctx, void *state) {
        static_cast<MitosisSteppable *>(state)->start(ctx);
    }

    void step(CC3DKernelContext *ctx, void *state) {
        static_cast<MitosisSteppable *>(state)->step(ctx);
    }

    void finish(CC3DKernelContext *ctx, void *state) {
        static_cast<MitosisSteppable *>(state)->finish(ctx);
    }

    void destroy(void *state) {
        delete static_cast<MitosisSteppable *>(state);
    }

    const CC3DSteppableV1 api = {
        CC3D_KERNEL_ABI_VERSION, &create, &start, &step, &finish, &destroy
    };
}

extern "C" CC3D_STEPPABLE_EXPORT const CC3DSteppableV1 *cc3d_get_steppable_v1() {
    return &api;
}
