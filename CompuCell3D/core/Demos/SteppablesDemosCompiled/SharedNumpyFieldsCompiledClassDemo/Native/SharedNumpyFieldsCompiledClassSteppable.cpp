#include <cc3d/kernel.h>

#include <iostream>

namespace {

class SharedNumpyFieldsCompiledSteppable {
public:
    void init(CC3DKernelContext *ctx) {
        scalar_ = ctx->sharedScalarFields.createScalar(
            "cpp_scalar", CC3D_FIELD_FLOAT32, 0);
        integer_ = ctx->sharedScalarFields.createScalar(
            "cpp_int16", CC3D_FIELD_INT16, 0);
        vector_ = ctx->sharedVectorFields.createVector(
            "cpp_vector", CC3D_FIELD_FLOAT32);
    }

    void start(CC3DKernelContext *) {
        scalar_.set(10, 20, 0, 42.0);
        integer_.set(10, 20, 0, 7.0);
        vector_.set(10, 20, 0, 1.0, 2.0, 3.0);

        std::cout << "C++ class shared fields created scalar="
                  << scalar_.get(10, 20, 0)
                  << " int16=" << integer_.get(10, 20, 0)
                  << std::endl;
    }

    void step(CC3DKernelContext *ctx) {
        double vx = 0.0;
        double vy = 0.0;
        double vz = 0.0;
        vector_.get(10, 20, 0, vx, vy, vz);

        if (ctx->mcs % 10 == 0) {
            int sampledPixels = 0;
            for (auto point : ctx->cellField.everyPixelWithSteps(20, 20, 1)) {
                (void)point;
                ++sampledPixels;
            }

            std::cout << "C++ class shared fields mcs=" << ctx->mcs
                      << " scalar=" << scalar_.get(15, 25, 0)
                      << " int16=" << integer_.get(15, 25, 0)
                      << " vector=(" << vx << "," << vy << "," << vz << ")"
                      << " sampled_pixels=" << sampledPixels
                      << std::endl;
        }
    }

    void finish(CC3DKernelContext *) {}

private:
    cc3d::kernel::SharedScalarField scalar_;
    cc3d::kernel::SharedScalarField integer_;
    cc3d::kernel::SharedVectorField vector_;
};

void *create() {
    return new SharedNumpyFieldsCompiledSteppable();
}

void init(CC3DKernelContext *ctx, void *state) {
    static_cast<SharedNumpyFieldsCompiledSteppable *>(state)->init(ctx);
}

void start(CC3DKernelContext *ctx, void *state) {
    static_cast<SharedNumpyFieldsCompiledSteppable *>(state)->start(ctx);
}

void step(CC3DKernelContext *ctx, void *state) {
    static_cast<SharedNumpyFieldsCompiledSteppable *>(state)->step(ctx);
}

void finish(CC3DKernelContext *ctx, void *state) {
    static_cast<SharedNumpyFieldsCompiledSteppable *>(state)->finish(ctx);
}

void destroy(void *state) {
    delete static_cast<SharedNumpyFieldsCompiledSteppable *>(state);
}

const CC3DSteppableV1 api = {
    CC3D_KERNEL_ABI_VERSION,
    &create,
    &start,
    &step,
    &finish,
    &destroy,
    &init
};

} // namespace

extern "C" CC3D_STEPPABLE_EXPORT
const CC3DSteppableV1 *cc3d_get_steppable_v1() {
    return &api;
}
