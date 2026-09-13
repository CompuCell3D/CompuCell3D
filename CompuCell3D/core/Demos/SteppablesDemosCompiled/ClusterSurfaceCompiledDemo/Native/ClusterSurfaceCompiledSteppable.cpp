#include <cc3d/kernel.h>

#include <iostream>
#include <set>

namespace {
    constexpr float TARGET_CLUSTER_SURFACE = 80.0f;
    constexpr float LAMBDA_CLUSTER_SURFACE = 2.0f;

    struct ClusterSurfaceState {
        unsigned long long calls = 0;
    };

    void *create() {
        return new ClusterSurfaceState;
    }

    void start(CC3DKernelContext *ctx, void *) {
        for (auto cell : ctx->cells) {
            cell.targetVolume = 25.0f;
            cell.lambdaVolume = 2.0f;
        }
        std::cerr << "Compiled ClusterSurface start\n";
    }

    void step(CC3DKernelContext *ctx, void *statePtr) {
        auto *state = static_cast<ClusterSurfaceState *>(statePtr);
        auto clusterSurface = ctx->clusterSurface.clusterSurface();
        std::set<long> updatedClusters;
        unsigned int printed = 0;

        for (auto cell : ctx->cells) {
            if (updatedClusters.insert(cell.clusterId).second) {
                clusterSurface.setTargetAndLambdaClusterSurface(
                        cell, TARGET_CLUSTER_SURFACE, LAMBDA_CLUSTER_SURFACE);
            }
        }

        for (auto cell : ctx->cells) {
            const auto parameters = clusterSurface.getTargetAndLambdaClusterSurface(cell);
            if (printed < 5) {
                std::cerr << "ClusterSurface cell=" << static_cast<long>(cell.id)
                          << " cluster=" << static_cast<long>(cell.clusterId)
                          << " target=" << parameters.first
                          << " lambda=" << parameters.second << '\n';
                ++printed;
            }
        }

        ++state->calls;
        std::cerr << "Compiled ClusterSurface step mcs=" << ctx->mcs
                  << " clusters=" << updatedClusters.size() << '\n';
    }

    void finish(CC3DKernelContext *ctx, void *statePtr) {
        const auto *state = static_cast<const ClusterSurfaceState *>(statePtr);
        std::cerr << "Compiled ClusterSurface finish mcs=" << ctx->mcs
                  << " calls=" << state->calls << '\n';
    }

    void destroy(void *statePtr) {
        delete static_cast<ClusterSurfaceState *>(statePtr);
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
