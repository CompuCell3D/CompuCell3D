#include <cc3d/kernel.h>

#include <algorithm>
#include <iostream>
#include <vector>

namespace {
    constexpr float INITIAL_TARGET_VOLUME = 25.0f;
    constexpr float LAMBDA_VOLUME = 2.0f;

    struct State {};
    void *create() { return new State; }
    void start(CC3DKernelContext *ctx, void *) {
        unsigned int cellCount = 0;
        for (auto cell : ctx->cells) {
            cell.targetVolume = INITIAL_TARGET_VOLUME;
            cell.lambdaVolume = LAMBDA_VOLUME;
            ++cellCount;
        }
        std::cerr << "Compiled periodic cluster mitosis start cells=" << cellCount
                  << " targetVolume=" << INITIAL_TARGET_VOLUME
                  << " lambdaVolume=" << LAMBDA_VOLUME << '\n';
    }
    void step(CC3DKernelContext *ctx, void *statePtr) {

        static_cast<void>(statePtr);
        for (auto cell : ctx->cells) cell.targetVolume += 1.0f;
        if (ctx->mcs < 20) return;


        auto mitosis = ctx->mitosis.mitosis();
        std::vector<long> clustersToDivide;
        for (auto cluster : ctx->cells.clusters()) {
            long clusterVolume = 0;
            for (auto cell : cluster.cells()) clusterVolume += cell.volume;
            std::cerr << "cluster_volume=" << clusterVolume
                      << " cluster_id=" << cluster.id() << '\n';
            if (clusterVolume > 250) clustersToDivide.push_back(cluster.id());
        }

        for (long clusterId : clustersToDivide) {
            mitosis.setParentChildPositionFlag(1);
            if (!mitosis.divideClusterOrientation(clusterId, 1.0, 0.0, 0.0)) continue;

            auto parent = mitosis.parentCell();
            auto child = mitosis.childCell();
            std::vector<cc3d::kernel::Cell> parentCells;
            std::vector<cc3d::kernel::Cell> childCells;
            for (auto cell : ctx->cells.clusterCells(parent.clusterId)) {
                cell.targetVolume = static_cast<float>(cell.targetVolume) / 2.0f;
                parentCells.push_back(cell);
            }
            for (auto cell : ctx->cells.clusterCells(child.clusterId)) {
                childCells.push_back(cell);
            }
            const auto count = std::min(parentCells.size(), childCells.size());
            for (size_t i = 0; i < count; ++i) {
                mitosis.cloneAttributes(parentCells[i], childCells[i]);
            }

            std::cerr << "periodic cluster division mcs=" << ctx->mcs
                      << " parentCluster=" << static_cast<long>(parent.clusterId)
                      << " childCluster=" << static_cast<long>(child.clusterId) << '\n';
        }
    }
    void finish(CC3DKernelContext *, void *) { std::cerr << "Compiled periodic cluster mitosis finish\n"; }
    void destroy(void *p) { delete static_cast<State *>(p); }
    const CC3DSteppableV1 api = {CC3D_KERNEL_ABI_VERSION, &create, &start, &step, &finish, &destroy};
}
extern "C" CC3D_STEPPABLE_EXPORT const CC3DSteppableV1 *cc3d_get_steppable_v1() { return &api; }
