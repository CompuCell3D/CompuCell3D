#include <cc3d/kernel.h>

#include <iostream>

namespace {
    struct State { bool divided = false; };
    void *create() { return new State; }
    void start(CC3DKernelContext *, void *) { std::cerr << "Compiled cluster mitosis start\n"; }
    void step(CC3DKernelContext *ctx, void *statePtr) {
        auto *state = static_cast<State *>(statePtr);
        if (state->divided) return;
        auto mitosis = ctx->mitosis.mitosis();
        long clusterId = -1;
        long clusterVolume = 0;
        for (auto cell : ctx->cells) {
            if (clusterId < 0) clusterId = cell.clusterId;
            if (cell.clusterId == clusterId) clusterVolume += cell.volume;
        }
        if (clusterId >= 0 && clusterVolume > 100) {
            mitosis.setParentChildPositionFlag(1);
            if (mitosis.divideClusterRandom(clusterId)) {
                auto parent = mitosis.parentCell();
                auto child = mitosis.childCell();
                mitosis.cloneAttributes(parent, child);
                std::cerr << "cluster division mcs=" << ctx->mcs
                          << " parentCluster=" << static_cast<long>(parent.clusterId)
                          << " childCluster=" << static_cast<long>(child.clusterId) << '\n';
                state->divided = true;
            }
        }
    }
    void finish(CC3DKernelContext *, void *) { std::cerr << "Compiled cluster mitosis finish\n"; }
    void destroy(void *p) { delete static_cast<State *>(p); }
    const CC3DSteppableV1 api = {CC3D_KERNEL_ABI_VERSION, &create, &start, &step, &finish, &destroy};
}
extern "C" CC3D_STEPPABLE_EXPORT const CC3DSteppableV1 *cc3d_get_steppable_v1() { return &api; }
