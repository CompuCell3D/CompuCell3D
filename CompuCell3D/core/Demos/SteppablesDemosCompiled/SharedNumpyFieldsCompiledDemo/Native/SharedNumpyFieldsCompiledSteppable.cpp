#include <cc3d/kernel.h>

#include <iostream>

namespace {
    struct State {
        cc3d::kernel::SharedScalarField scalar;
        cc3d::kernel::SharedScalarField integer;
        cc3d::kernel::SharedVectorField vector;

    };

    void *create() { return new State; }

    void init(CC3DKernelContext *ctx, void *statePtr) {
        auto *state = static_cast<State *>(statePtr);
        state->scalar = ctx->sharedScalarFields.createScalar("cpp_scalar", CC3D_FIELD_FLOAT32, 0);
        state->integer = ctx->sharedScalarFields.createScalar("cpp_int16", CC3D_FIELD_INT16, 0);
        state->vector = ctx->sharedVectorFields.createVector("cpp_vector", CC3D_FIELD_FLOAT32);
    }

    void start(CC3DKernelContext *ctx, void *statePtr) {
        auto *state = static_cast<State *>(statePtr);

        state->scalar.set(10, 20, 0, 42.0);
        state->integer.set(10, 20, 0, 7.0);
        state->vector.set(10, 20, 0, 1.0, 2.0, 3.0);

        std::cout << "C++ shared fields created scalar=" << state->scalar.get(10, 20, 0)
                  << " int16=" << state->integer.get(10, 20, 0)
                  << std::endl;
    }

    void step(CC3DKernelContext *ctx, void *statePtr) {
        auto *state = static_cast<State *>(statePtr);
        double vx = 0.0, vy = 0.0, vz = 0.0;
        state->vector.get(10, 20, 0, vx, vy, vz);

        if (ctx->mcs % 10 == 0) {
            int sampledPixels = 0;
            for (auto point : ctx->cellField.everyPixelWithSteps(20, 20, 1)) {
                (void) point;
                ++sampledPixels;
            }

            std::cout << "C++ shared fields mcs=" << ctx->mcs
                      << " scalar=" << state->scalar.get(15, 25, 0)
                      << " int16=" << state->integer.get(15, 25, 0)
                      << " vector=(" << vx << "," << vy << "," << vz << ")"
                      << " sampled_pixels=" << sampledPixels
                      << std::endl;
        }
    }

    void finish(CC3DKernelContext *, void *) {}
    void destroy(void *state) { delete static_cast<State *>(state); }

    const CC3DSteppableV1 api = {
        CC3D_KERNEL_ABI_VERSION, &create, &start, &step, &finish, &destroy, &init
    };
}

extern "C" CC3D_STEPPABLE_EXPORT const CC3DSteppableV1 *cc3d_get_steppable_v1() {
    return &api;
}
