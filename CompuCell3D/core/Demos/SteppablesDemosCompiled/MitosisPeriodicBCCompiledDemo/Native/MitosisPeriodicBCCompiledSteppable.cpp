#include <cc3d/kernel.h>

#include <iostream>

namespace {
    struct State { bool divided = false; };
    void *create() { return new State; }
    void start(CC3DKernelContext *, void *) { std::cerr << "Compiled periodic cluster mitosis start\n"; }
    void step(CC3DKernelContext *ctx, void *statePtr) {
        auto *state = static_cast<State *>(statePtr);
        if (state->divided || ctx->mcs < 20) return;
        auto mitosis = ctx->mitosis.mitosis();
        long clusterId = -1;
        for (auto cell : ctx->cells) { clusterId = cell.clusterId; break; }
        if (clusterId >= 0 && mitosis.divideClusterOrientation(clusterId, 1.0, 0.0, 0.0)) {
            auto parent = mitosis.parentCell();
            auto child = mitosis.childCell();
            mitosis.cloneAttributes(parent, child);
            std::cerr << "periodic cluster division mcs=" << ctx->mcs
                      << " parent=" << static_cast<long>(parent.id)
                      << " child=" << static_cast<long>(child.id) << '\n';
            state->divided = true;
        }
    }
    void finish(CC3DKernelContext *, void *) { std::cerr << "Compiled periodic cluster mitosis finish\n"; }
    void destroy(void *p) { delete static_cast<State *>(p); }
    const CC3DSteppableV1 api = {CC3D_KERNEL_ABI_VERSION, &create, &start, &step, &finish, &destroy};
}
extern "C" CC3D_STEPPABLE_EXPORT const CC3DSteppableV1 *cc3d_get_steppable_v1() { return &api; }
