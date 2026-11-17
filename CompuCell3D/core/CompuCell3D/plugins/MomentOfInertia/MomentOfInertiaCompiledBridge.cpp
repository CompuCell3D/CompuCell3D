#include "MomentOfInertiaPlugin.h"
#include <cc3d/CompiledSteppablePluginAPI.h>

namespace {
using Getter = std::vector<double> (CompuCell3D::MomentOfInertiaPlugin::*)(CompuCell3D::CellG *);

size_t get_values(void *instance, CC3DPluginCellHandle cell, double *values, size_t capacity, Getter getter) {
    if (!values || !cell) return 0u;
    const auto result = (static_cast<CompuCell3D::MomentOfInertiaPlugin *>(instance)->*getter)(
            static_cast<CompuCell3D::CellG *>(cell));
    const size_t count = result.size() < capacity ? result.size() : capacity;
    for (size_t i = 0; i < count; ++i) values[i] = result[i];
    return result.size();
}

size_t semiaxes(void *i, CC3DPluginCellHandle c, double *v, size_t n) { return get_values(i, c, v, n, &CompuCell3D::MomentOfInertiaPlugin::getSemiaxes); }
size_t semiaxes_xy(void *i, CC3DPluginCellHandle c, double *v, size_t n) { return get_values(i, c, v, n, &CompuCell3D::MomentOfInertiaPlugin::getSemiaxesXY); }
size_t semiaxes_xz(void *i, CC3DPluginCellHandle c, double *v, size_t n) { return get_values(i, c, v, n, &CompuCell3D::MomentOfInertiaPlugin::getSemiaxesXZ); }
size_t semiaxes_yz(void *i, CC3DPluginCellHandle c, double *v, size_t n) { return get_values(i, c, v, n, &CompuCell3D::MomentOfInertiaPlugin::getSemiaxesYZ); }
size_t semiaxes_3d(void *i, CC3DPluginCellHandle c, double *v, size_t n) { return get_values(i, c, v, n, &CompuCell3D::MomentOfInertiaPlugin::getSemiaxes3D); }
}

extern "C" void cc3d_register_moment_of_inertia_plugin(void *simulator, void *plugin) {
    CC3DMomentOfInertiaCallbacksV1 callbacks{};
    callbacks.plugin_instance = plugin;
    callbacks.get_semiaxes_fn = &semiaxes;
    callbacks.get_semiaxes_xy_fn = &semiaxes_xy;
    callbacks.get_semiaxes_xz_fn = &semiaxes_xz;
    callbacks.get_semiaxes_yz_fn = &semiaxes_yz;
    callbacks.get_semiaxes_3d_fn = &semiaxes_3d;
    cc3d_register_moment_of_inertia_callbacks(simulator, &callbacks);
}
