#include "ClusterSurfacePlugin.h"
#include <cc3d/CompiledSteppablePluginAPI.h>

namespace {
void set_cluster_surface(void *instance, CC3DPluginCellHandle cell, float target, float lambda) {
    static_cast<CompuCell3D::ClusterSurfacePlugin *>(instance)->setTargetAndLambdaClusterSurface(
            static_cast<CompuCell3D::CellG *>(cell), target, lambda);
}

void get_cluster_surface(void *instance, CC3DPluginCellHandle cell, float *target, float *lambda) {
    if (!target || !lambda) return;
    const auto values = static_cast<CompuCell3D::ClusterSurfacePlugin *>(instance)->getTargetAndLambdaClusterSurface(
            static_cast<const CompuCell3D::CellG *>(cell));
    *target = values.first;
    *lambda = values.second;
}
}

extern "C" void cc3d_register_cluster_surface_plugin(void *simulator, void *plugin) {
    CC3DClusterSurfaceCallbacksV1 callbacks{};
    callbacks.plugin_instance = plugin;
    callbacks.set_fn = &set_cluster_surface;
    callbacks.get_fn = &get_cluster_surface;
    cc3d_register_cluster_surface_callbacks(simulator, &callbacks);
}
