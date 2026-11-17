// Keep this demo on the repository C ABI header so its cluster iterators are
// available even when the Python environment has an older installed header.
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
        std::cerr << "Compiled cluster mitosis start cells=" << cellCount
                  << " targetVolume=" << INITIAL_TARGET_VOLUME
                  << " lambdaVolume=" << LAMBDA_VOLUME << '\n';
    }
    void step(CC3DKernelContext *ctx, void *statePtr) {
        static_cast<void>(statePtr);

        // Match the Python cluster-mitosis demo's VolumeParamSteppable.
        for (auto cell : ctx->cells) cell.targetVolume += 1.0f;

        // Match get_cluster_cells() and the diagnostic loop in the Python demo.
        for (auto cell : ctx->cells) {
            std::cerr << "DISPLAYING CELL IDS OF CLUSTER "
                      << static_cast<long>(cell.clusterId)
                      << " CELL. ID=" << static_cast<long>(cell.id) << '\n';
            for (auto clusterCell : ctx->cells.clusterCells(cell.clusterId)) {
                std::cerr << "CLUSTER CELL ID=" << static_cast<long>(clusterCell.id)
                          << " type=" << static_cast<unsigned int>(clusterCell.type) << '\n';
            }
        }

        auto mitosis = ctx->mitosis.mitosis();
        std::vector<long> clustersToDivide;
        for (auto cluster : ctx->cells.clusters()) {
            long clusterVolume = 0;
            for (auto cell : cluster.cells()) {
                clusterVolume += cell.volume;
            }
            std::cerr << "cluster_volume=" << clusterVolume
                      << " cluster_id=" << cluster.id() << '\n';
            if (clusterVolume > 250) {
                clustersToDivide.push_back(cluster.id());
            }
        }

        for (long clusterId : clustersToDivide) {
            mitosis.setParentChildPositionFlag(1);
            if (mitosis.divideClusterRandom(clusterId)) {
                auto parent = mitosis.parentCell();
                auto child = mitosis.childCell();

                // The parent and child cluster compartments are returned in
                // matching order, just as in clone_cluster_attributes().
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

                std::cerr << "cluster division mcs=" << ctx->mcs
                          << " parentCluster=" << static_cast<long>(parent.clusterId)
                          << " childCluster=" << static_cast<long>(child.clusterId) << '\n';
            }
        }
    }
    void finish(CC3DKernelContext *, void *) { std::cerr << "Compiled cluster mitosis finish\n"; }
    void destroy(void *p) { delete static_cast<State *>(p); }
    const CC3DSteppableV1 api = {CC3D_KERNEL_ABI_VERSION, &create, &start, &step, &finish, &destroy};
}
extern "C" CC3D_STEPPABLE_EXPORT const CC3DSteppableV1 *cc3d_get_steppable_v1() { return &api; }
