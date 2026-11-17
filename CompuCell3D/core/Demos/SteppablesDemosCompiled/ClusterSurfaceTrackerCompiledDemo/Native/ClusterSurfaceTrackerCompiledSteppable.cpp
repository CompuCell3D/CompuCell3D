#include <cc3d/kernel.h>

#include <iostream>

namespace {
    struct ClusterSurfaceTrackerState {
        bool initialized = false;
        unsigned long long calls = 0;
    };

    void *create() {
        return new ClusterSurfaceTrackerState;
    }

    void start(CC3DKernelContext *, void *) {
        std::cerr << "Compiled ClusterSurfaceTracker start\n";
    }

    void step(CC3DKernelContext *ctx, void *statePtr) {
        auto *state = static_cast<ClusterSurfaceTrackerState *>(statePtr);
        auto tracker = ctx->clusterSurfaceTracker.clusterSurfaceTracker();

        if (!state->initialized) {
            long clusterId = -1;
            for (int i = 0; i < 6; ++i) {
                auto cell = ctx->newCell(static_cast<unsigned char>(i == 0 || i == 5 ? 1 : 2));
                if (i == 0) {
                    clusterId = cell.clusterId;
                } else {
                    tracker.reassignClusterId(cell, clusterId);
                }
                ctx->cellField.assign(25, 25 + 5, 25 + i * 5, 25 + (i + 1) * 5, 0, 1, cell);
            }
            tracker.updateClusterSurface(clusterId);
            state->initialized = true;
            std::cerr << "Created elongated cluster id=" << clusterId << '\n';
        }

        unsigned int printed = 0;
        for (auto cell : ctx->cells) {
            if (printed++ < 6) {
                std::cerr << "ClusterSurfaceTracker cell=" << static_cast<long>(cell.id)
                          << " cluster=" << static_cast<long>(cell.clusterId)
                          << " volume=" << static_cast<long>(cell.volume)
                          << " surface=" << static_cast<double>(cell.clusterSurface) << '\n';
            }
        }
        ++state->calls;
    }

    void finish(CC3DKernelContext *ctx, void *statePtr) {
        const auto *state = static_cast<const ClusterSurfaceTrackerState *>(statePtr);
        std::cerr << "Compiled ClusterSurfaceTracker finish mcs=" << ctx->mcs
                  << " calls=" << state->calls << '\n';
    }

    void destroy(void *statePtr) {
        delete static_cast<ClusterSurfaceTrackerState *>(statePtr);
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
