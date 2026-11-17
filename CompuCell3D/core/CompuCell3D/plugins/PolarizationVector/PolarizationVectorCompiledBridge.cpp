#include "PolarizationVectorPlugin.h"
#include <cc3d/CompiledSteppablePluginAPI.h>

namespace {
void set_vector(void *instance, CC3DPluginCellHandle cell, float x, float y, float z) {
    static_cast<CompuCell3D::PolarizationVectorPlugin *>(instance)->setPolarizationVector(static_cast<CompuCell3D::CellG *>(cell), x, y, z);
}
uint8_t get_vector(void *instance, CC3DPluginCellHandle cell, float *x, float *y, float *z) {
    if (!x || !y || !z) return 0;
    auto values = static_cast<CompuCell3D::PolarizationVectorPlugin *>(instance)->getPolarizationVector(static_cast<CompuCell3D::CellG *>(cell));
    if (values.size() != 3) return 0;
    *x = values[0]; *y = values[1]; *z = values[2];
    return 1;
}
}

extern "C" void cc3d_register_polarization_vector_plugin(void *simulator, void *plugin) {
    CC3DPolarizationVectorCallbacksV1 callbacks{};
    callbacks.plugin_instance = plugin;
    callbacks.set_fn = &set_vector;
    callbacks.get_fn = &get_vector;
    cc3d_register_polarization_vector_callbacks(simulator, &callbacks);
}
