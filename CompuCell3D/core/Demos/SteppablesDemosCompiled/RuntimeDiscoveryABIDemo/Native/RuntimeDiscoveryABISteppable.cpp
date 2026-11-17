#include <cc3d/kernel.h>

#include <iostream>
#include <string>

namespace {
    struct State { bool checked = false; };

    void *create() { return new State; }

    void check(CC3DKernelContext *ctx, State *state) {
        auto shared = ctx->sharedData.sharedData(&ctx->dictionaryQueries);
        int failures = 0;
        auto checkFlag = [&](const char *expectedKey, bool actual) {
            bool expected = false;
            if (!shared.getBool(expectedKey, expected) || expected != actual) ++failures;
        };

        checkFlag("python_plugin_cell_type_loaded", ctx->discovery.hasPlugin("CellType"));
        checkFlag("python_plugin_missing_loaded", ctx->discovery.hasPlugin("DefinitelyMissingPlugin"));
        checkFlag("python_plugin_cell_type_registered", ctx->discovery.isPluginRegistered("CellType"));
        checkFlag("python_steppable_blob_loaded", ctx->discovery.hasSteppable("BlobInitializer"));
        checkFlag("python_steppable_missing_loaded", ctx->discovery.hasSteppable("DefinitelyMissingSteppable"));
        checkFlag("python_steppable_blob_registered", ctx->discovery.isSteppableRegistered("BlobInitializer"));

        const auto plugins = ctx->discovery.loadedPlugins();
        const auto steppables = ctx->discovery.loadedSteppables();
        std::cerr << "Runtime discovery ABI plugins:";
        for (const auto &name : plugins) std::cerr << " " << name;
        std::cerr << "\nRuntime discovery ABI steppables:";
        for (const auto &name : steppables) std::cerr << " " << name;
        std::cerr << "\n";

        shared.setInt("discovery_cpp_failures", failures);
        shared.setInt("discovery_cpp_plugin_count", static_cast<int64_t>(plugins.size()));
        shared.setInt("discovery_cpp_steppable_count", static_cast<int64_t>(steppables.size()));
        state->checked = true;
    }

    void start(CC3DKernelContext *, void *) {}
    void step(CC3DKernelContext *ctx, void *statePtr) {
        auto *state = static_cast<State *>(statePtr);
        if (!state->checked) check(ctx, state);
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
