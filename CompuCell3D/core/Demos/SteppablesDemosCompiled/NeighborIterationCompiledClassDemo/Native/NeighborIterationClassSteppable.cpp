#include <cc3d/kernel.h>
#include <iostream>
#include <cstdio>
#include <exception>

namespace {
    constexpr unsigned char TYPE_A = 1u;
    constexpr unsigned char TYPE_B = 2u;

    class NeighborIterationSteppable {
    public:
        void start(CC3DKernelContext *ctx) {
            executionCount_ = 0u;
            neighborTrackerAvailable_ = false;
            neighborTrackerChecked_ = false;
            seed_cells(ctx);
            std::printf("Compiled neighbor iteration start\n");
        }

        void step(CC3DKernelContext *ctx) {
            try {
                neighborTrackerAvailable_ = print_neighbor_snapshot(ctx);
                neighborTrackerChecked_ = true;
            } catch (const std::exception &e) {
                if (!neighborTrackerChecked_ || neighborTrackerAvailable_) {
                    std::printf("Compiled neighbor iteration unavailable: %s\n", e.what());
                }
                neighborTrackerAvailable_ = false;
                neighborTrackerChecked_ = true;
            }

            ++executionCount_;
            std::printf(
                "Compiled neighbor iteration step mcs=%llu execution=%llu tracker=%s\n",
                static_cast<unsigned long long>(ctx->mcs),
                static_cast<unsigned long long>(executionCount_),
                neighborTrackerAvailable_ ? "available" : "missing"
            );
        }

        void finish(CC3DKernelContext *ctx) {
            std::printf(
                "Compiled neighbor iteration finish mcs=%llu execution=%llu tracker_checked=%s\n",
                static_cast<unsigned long long>(ctx->mcs),
                static_cast<unsigned long long>(executionCount_),
                neighborTrackerChecked_ ? "yes" : "no"
            );
        }

    private:
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
            int sampledCells = 0;
            for (auto cell : ctx->cells) {
                int totalNeighbors = 0;
                int mediumContacts = 0;
                int cellContacts = 0;
                int sharedSurface = 0;
                std::cerr << "MCS=" << ctx->mcs << std::endl;
                for (auto neighborInfo : ctx->neighbors.forCell(cell)) {
                    ++totalNeighbors;
                    sharedSurface += neighborInfo.commonSurfaceArea;
                    if (neighborInfo) {
                        ++cellContacts;
                        auto neighbor = neighborInfo.neighbor();
                        std::cerr << "cell.id=" << cell.id
                                  << " cell.type=" << cell.type
                                  << " neighbor.id=" << neighbor.id
                                  << " common surface area="
                                  << neighborInfo.commonSurfaceArea << std::endl;
                    } else {
                        ++mediumContacts;
                        std::cerr << "cell.id=" << cell.id
                                  << " cell.type=" << cell.type
                                  << " neighbor.id=MEDIUM"
                                  << " common surface area="
                                  << neighborInfo.commonSurfaceArea << std::endl;
                    }
                }
                std::cerr << "Compiled neighbor summary cell.id=" << cell.id
                          << " total_neighbors=" << totalNeighbors
                          << " cell_neighbors=" << cellContacts
                          << " mediumContacts=" << mediumContacts
                          << " sharedSurface=" << sharedSurface << std::endl;

                if (++sampledCells >= 2) break;
            }
            return true;
        }

        uint64_t executionCount_ = 0u;
        bool neighborTrackerAvailable_ = false;
        bool neighborTrackerChecked_ = false;
    };

    void *create() { return new NeighborIterationSteppable(); }

    void start(CC3DKernelContext *ctx, void *state) {
        static_cast<NeighborIterationSteppable *>(state)->start(ctx);
    }

    void step(CC3DKernelContext *ctx, void *state) {
        static_cast<NeighborIterationSteppable *>(state)->step(ctx);
    }

    void finish(CC3DKernelContext *ctx, void *state) {
        static_cast<NeighborIterationSteppable *>(state)->finish(ctx);
    }

    void destroy(void *state) {
        delete static_cast<NeighborIterationSteppable *>(state);
    }

    const CC3DSteppableV1 api = {
        CC3D_KERNEL_ABI_VERSION, &create, &start, &step, &finish, &destroy
    };
}

extern "C" CC3D_STEPPABLE_EXPORT const CC3DSteppableV1 *cc3d_get_steppable_v1() {
    return &api;
}
