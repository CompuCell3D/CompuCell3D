#include <cc3d/kernel.h>

#include <iostream>
#include <vector>

namespace {
    struct State { unsigned long long steps = 0; };

    void *create() { return new State; }

    void start(CC3DKernelContext *ctx, void *) {
        auto shared = ctx->sharedData.sharedData();
        shared.setString("cpp_status", "started");
        shared.setInt("cpp_step_count", 0);

        const int cellType = ctx->cellTypes["Cell"];
        if (cellType >= 0) {
            for (auto cell : ctx->cells.byType(static_cast<unsigned char>(cellType))) {
                auto attributes = ctx->cellAttributes.forCell(cell);
                attributes.setString("cpp_named_type", ctx->cellTypes.name(cell).c_str());
            }
        }

        auto firstCell = ctx->cells.fetch_cell_by_id(1);
        if (firstCell) {
            auto attributes = ctx->cellAttributes.forCell(firstCell.value());
            attributes.setBool("cpp_lookup_by_id", true);
        }

        for (auto cell : ctx->cells) {
            cell.targetVolume=25.;
            cell.lambdaVolume=2.;

        }

        for (auto cell : ctx->cells) {
            auto attributes = ctx->cellAttributes.forCell(cell);
            attributes.setString("cpp_status", "ready");
            attributes.setTuple("cpp_vector", std::vector<double>{1.0, 2.0, 3.0});
        }
    }

    void step(CC3DKernelContext *ctx, void *statePtr) {
        auto *state = static_cast<State *>(statePtr);
        auto shared = ctx->sharedData.sharedData();

        double pythonValue = 0.0;
        if (shared.getDouble("python_to_cpp", pythonValue)) {
            shared.setDouble("cpp_seen_python_value", pythonValue);
        }
        shared.setInt("cpp_step_count", static_cast<int64_t>(++state->steps));
        shared.setDouble("cpp_to_python", static_cast<double>(ctx->mcs) + 0.5);

        for (auto cell : ctx->cells) {
            auto attributes = ctx->cellAttributes.forCell(cell);
            double pythonSignal = 0.0;
            const auto pythonVector = attributes.getTuple("python_vector");
            if (attributes.getDouble("python_signal", pythonSignal)) {
                attributes.setDouble("cpp_seen_signal", pythonSignal);
            }
            if (!pythonVector.empty()) {
                attributes.setTuple("cpp_seen_vector", pythonVector);
            }
        }

        if (ctx->mcs % 10 == 0) {
            std::cerr << "C++ exchange mcs=" << ctx->mcs
                      << " python_to_cpp=" << pythonValue
                      << " cpp_step_count=" << state->steps << '\n';
        }
    }

    void finish(CC3DKernelContext *, void *) {}
    void destroy(void *state) { delete static_cast<State *>(state); }

    const CC3DSteppableV1 api = {
        CC3D_KERNEL_ABI_VERSION, &create, &start, &step, &finish, &destroy
    };
}

extern "C" CC3D_STEPPABLE_EXPORT const CC3DSteppableV1 *cc3d_get_steppable_v1() {
    return &api;
}
