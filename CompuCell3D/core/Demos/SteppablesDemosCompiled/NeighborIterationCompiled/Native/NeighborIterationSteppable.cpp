#include <cc3d/kernel.h>
#include <iostream>
#include <cstdio>
#include <exception>

namespace {
    constexpr unsigned char TYPE_A = 1u;
    constexpr unsigned char TYPE_B = 2u;

    struct NeighborIterationState {
        uint64_t executionCount;
        bool neighborTrackerAvailable;
        bool neighborTrackerChecked;
    };

    void *create() {
        return new NeighborIterationState{0u, false, false};
    }

    void seed_cells(CC3DKernelContext *ctx) {
        auto left = ctx->newCell(TYPE_A);
        left.targetVolume = 36.0f;
        left.lambdaVolume = 2.0f;
        ctx->cellField.assign(8, 20, 12, 24, 0, 1, left);

        auto right = ctx->newCell(TYPE_B);
        right.targetVolume = 36.0f;
        right.lambdaVolume = 2.0f;
        ctx->cellField.assign(20, 32, 12, 24, 0, 1, right);
    }

    bool print_neighbor_snapshot(CC3DKernelContext *ctx) {
        using namespace std;
        int sampledCells = 0;
        for (auto cell : ctx->cells) {
            int totalNeighbors = 0;
            int mediumContacts = 0;
            int cellContacts = 0;
            int sharedSurface = 0;
            cerr<<"MCS="<<ctx->mcs<<endl;
            for (auto neighborInfo : ctx->neighbors.forCell(cell)) {
                ++totalNeighbors;
                sharedSurface += neighborInfo.commonSurfaceArea;
                if (neighborInfo) {
                    ++cellContacts;
                    auto neighbor = neighborInfo.neighbor();
                    cerr<<"cell.id="<<cell.id<<" cell.type="<<cell.type<<" neighbor.id="<<neighbor.id<<" common surface area="<<neighborInfo.commonSurfaceArea<<endl;

                } else {
                    ++mediumContacts;
                    cerr<<"cell.id="<<cell.id<<" cell.type="<<cell.type<<" neighbor.id=MEDIUM"<<" common surface area="<<neighborInfo.commonSurfaceArea<<endl;

                }
            }
            cerr<<"Compiled neighbor summary cell.id="<<cell.id<<" total_neighbors="<<totalNeighbors<<" cell_neighbors="<<cellContacts<<" mediumContacts="<<mediumContacts<<" sharedSurface="<<sharedSurface<<endl;

            if (++sampledCells >= 2) {
                break;
            }
        }

        return true;
    }

    void start(CC3DKernelContext *ctx, void *statePtr) {
        auto *state = static_cast<NeighborIterationState *>(statePtr);
        state->executionCount = 0u;
        state->neighborTrackerAvailable = false;
        state->neighborTrackerChecked = false;

        seed_cells(ctx);
        std::printf("Compiled neighbor iteration start\n");
    }

    void step(CC3DKernelContext *ctx, void *statePtr) {
        auto *state = static_cast<NeighborIterationState *>(statePtr);

        try {
            state->neighborTrackerAvailable = print_neighbor_snapshot(ctx);
            state->neighborTrackerChecked = true;
        } catch (const std::exception &e) {
            if (!state->neighborTrackerChecked || state->neighborTrackerAvailable) {
                std::printf("Compiled neighbor iteration unavailable: %s\n", e.what());
            }
            state->neighborTrackerAvailable = false;
            state->neighborTrackerChecked = true;
        }

        ++state->executionCount;
        std::printf(
            "Compiled neighbor iteration step mcs=%llu execution=%llu tracker=%s\n",
            static_cast<unsigned long long>(ctx->mcs),
            static_cast<unsigned long long>(state->executionCount),
            state->neighborTrackerAvailable ? "available" : "missing"
        );
    }

    void finish(CC3DKernelContext *ctx, void *statePtr) {
        auto *state = static_cast<NeighborIterationState *>(statePtr);
        std::printf(
            "Compiled neighbor iteration finish mcs=%llu execution=%llu tracker_checked=%s\n",
            static_cast<unsigned long long>(ctx->mcs),
            static_cast<unsigned long long>(state->executionCount),
            state->neighborTrackerChecked ? "yes" : "no"
        );
    }

    void destroy(void *statePtr) {
        delete static_cast<NeighborIterationState *>(statePtr);
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

extern "C" CC3D_STEPPABLE_EXPORT const CC3DSteppableV1 *cc3d_get_steppable_v1() {
    return &api;
}
