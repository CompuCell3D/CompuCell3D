#include "CellOrientationPlugin.h"
#include <cc3d/CompiledSteppablePluginAPI.h>

namespace {
void set_lambda(void *instance, CC3DPluginCellHandle cell, double value) {
    static_cast<CompuCell3D::CellOrientationPlugin *>(instance)->setLambdaCellOrientation(
        static_cast<CompuCell3D::CellG *>(cell), value);
}

double get_lambda(void *instance, CC3DPluginCellHandle cell) {
    return static_cast<CompuCell3D::CellOrientationPlugin *>(instance)->getLambdaCellOrientation(
        static_cast<CompuCell3D::CellG *>(cell));
}
}

extern "C" void cc3d_register_cell_orientation_plugin(void *simulator, void *plugin) {
    CC3DCellOrientationCallbacksV1 callbacks{};
    callbacks.plugin_instance = plugin;
    callbacks.set_lambda_fn = &set_lambda;
    callbacks.get_lambda_fn = &get_lambda;
    cc3d_register_cell_orientation_callbacks(simulator, &callbacks);
}
