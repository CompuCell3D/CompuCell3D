#include "ClusterSurfaceTrackerPlugin.h"
#include <cc3d/CompiledSteppablePluginAPI.h>

namespace {
void update_cluster_surface(void *instance, long cluster_id) {
    static_cast<CompuCell3D::ClusterSurfaceTrackerPlugin *>(instance)->updateClusterSurface(cluster_id);
}
}

extern "C" void cc3d_register_cluster_surface_tracker_plugin(void *simulator, void *plugin) {
    CC3DClusterSurfaceTrackerCallbacksV1 callbacks{};
    callbacks.plugin_instance = plugin;
    callbacks.update_cluster_surface_fn = &update_cluster_surface;
    cc3d_register_cluster_surface_tracker_callbacks(simulator, &callbacks);
}
