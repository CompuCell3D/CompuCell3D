#include <cc3d/kernel.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {
    using Point = CC3DGeometryPointV1;

    bool close(double a, double b, double tolerance = 1.0e-9) {
        return std::abs(a - b) <= tolerance;
    }

    bool close(Point a, Point b) {
        return close(a.x, b.x) && close(a.y, b.y) && close(a.z, b.z);
    }

    struct State { bool checked = false; };
    void *create() { return new State; }

    void start(CC3DKernelContext *ctx, void *statePtr) {
        auto *state = static_cast<State *>(statePtr);
        auto shared = ctx->sharedData.sharedData(&ctx->dictionaryQueries);

        const Point p1{10.25, 12.5, 0.0};
        const Point p2{95.75, 70.0, 0.0};
        auto ordinary = ctx->geometry.distanceVector(p1, p2);
        auto invariant = ctx->geometry.invariantDistanceVector(p1, p2);
        auto unconditional = ctx->geometry.unconditionalInvariantDistanceVector(p1, p2);
        const Point hexPoint = ctx->geometry.hexToCartesian(Point{1.0, 2.0, 0.0});
        const Point cartPoint = ctx->geometry.cartesianToHex(Point{3.0, 4.0, 0.0});

        int failures = 0;
        double maxError = 0.0;
        auto checkScalar = [&](const char *key, double actual) {
            double wanted = 0.0;
            std::string fullKey = std::string("geometry_python_") + key;
            if (!shared.getDouble(fullKey.c_str(), wanted) || !close(actual, wanted)) ++failures;
            maxError = std::max(maxError, std::abs(actual - wanted));
        };
        auto checkVector = [&](const char *key, Point actual) {
            std::string fullKey = std::string("geometry_python_") + key;
            auto wanted = shared.getTuple(fullKey.c_str());
            if (wanted.size() != 3 || !close(actual.x, wanted[0]) || !close(actual.y, wanted[1]) || !close(actual.z, wanted[2])) ++failures;
            if (wanted.size() == 3) {
                maxError = std::max(maxError, std::abs(actual.x - wanted[0]));
                maxError = std::max(maxError, std::abs(actual.y - wanted[1]));
                maxError = std::max(maxError, std::abs(actual.z - wanted[2]));
            }
        };
        checkScalar("distance", ctx->geometry.distance(p1, p2));
        checkVector("ordinary_vector", ordinary);
        checkScalar("invariant_distance", ctx->geometry.invariantDistance(p1, p2));
        checkVector("invariant_vector", invariant);
        checkVector("unconditional_vector", unconditional);
        checkScalar("norm", ctx->geometry.vectorNorm(Point{3.0, 4.0, 12.0}));
        checkVector("hex_point", hexPoint);
        checkVector("cart_point", cartPoint);

        bool wantedBounds = false;
        bool bounds = ctx->geometry.isInBounds(Point{10.0, 10.0, 0.0});
        if (!shared.getBool("geometry_python_in_bounds", wantedBounds) || bounds != wantedBounds) ++failures;

        shared.setDouble("geometry_cpp_max_error", maxError);
        shared.setInt("geometry_cpp_failures", failures);
        state->checked = true;
        std::cerr << "Geometry ABI demo failures=" << failures << " max_error=" << maxError << '\n';
    }

    void step(CC3DKernelContext *ctx, void *statePtr) {
        auto *state = static_cast<State *>(statePtr);
        if (!state->checked) start(ctx, statePtr);
    }
    void finish(CC3DKernelContext *, void *) {}
    void destroy(void *state) { delete static_cast<State *>(state); }

    const CC3DSteppableV1 api = {
        CC3D_KERNEL_ABI_VERSION, &create, nullptr, &step, &finish, &destroy, nullptr
    };
}

extern "C" CC3D_STEPPABLE_EXPORT const CC3DSteppableV1 *cc3d_get_steppable_v1() {
    return &api;
}
