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
    struct State {
        unsigned long long divisions = 0;
        std::mt19937 random{0xCC3D2026u};
        bool missingFgfReported = false;
        bool initialized = false;
    };

    void *create() { return new State; }
    void initialize_cells(CC3DKernelContext *ctx, State *state) {
        unsigned int cells = 0;
        for (auto cell : ctx->cells) {
            cell.targetVolume = INITIAL_TARGET_VOLUME;
            cell.lambdaVolume = LAMBDA_VOLUME;
            ++cells;
        }
        if (cells == 0) return;
        state->initialized = true;
        std::cerr << "Compiled mitosis growth start cells=" << cells
                  << " targetVolume=" << INITIAL_TARGET_VOLUME
                  << " lambdaVolume=" << LAMBDA_VOLUME << '\n';
    }

    void start(CC3DKernelContext *ctx, void *statePtr) {
        initialize_cells(ctx, static_cast<State *>(statePtr));
    }

    void step(CC3DKernelContext *ctx, void *statePtr) {
        auto *state = static_cast<State *>(statePtr);
        if (!state->initialized) initialize_cells(ctx, state);
        auto fgf = ctx->scalarFields["FGF"];
        if (!fgf) {
            if (!state->missingFgfReported) {
                std::cerr << "Compiled mitosis growth: FGF field is unavailable\n";
                state->missingFgfReported = true;
            }
            return;
        }

        const auto dimensions = fgf.dim();
        std::vector<cc3d::kernel::Cell> cellsToDivide;
        for (auto cell : ctx->cells) {
            if (cell.type != CONDENSING) continue;

            int x = static_cast<int>(std::lround(static_cast<double>(cell.xCOM)));
            int y = static_cast<int>(std::lround(static_cast<double>(cell.yCOM)));
            int z = static_cast<int>(std::lround(static_cast<double>(cell.zCOM)));


            const float concentration = static_cast<float>(fgf(x, y, z));
            cell.targetVolume += 0.1f * concentration;

            // Initial target volume is 25, so 50 is the doubling-volume threshold.
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
            if (differentiate(state->random)) {
                child.type = CONDENSING_DIFFERENTIATED;
            }
            ++state->divisions;
            std::cerr << "mitosis mcs=" << ctx->mcs
                      << " source=" << parentId
                      << " parent=" << static_cast<long>(parent.id)
                      << " child=" << static_cast<long>(child.id)
                      << " childType=" << static_cast<unsigned int>(child.type) << '\n';
        }
    }
    void finish(CC3DKernelContext *, void *statePtr) {
        std::cerr << "Compiled single-cell mitosis divisions="
                  << static_cast<State *>(statePtr)->divisions << '\n';
    }
    void destroy(void *p) { delete static_cast<State *>(p); }
    const CC3DSteppableV1 api = {CC3D_KERNEL_ABI_VERSION, &create, &start, &step, &finish, &destroy};
}
extern "C" CC3D_STEPPABLE_EXPORT const CC3DSteppableV1 *cc3d_get_steppable_v1() { return &api; }
