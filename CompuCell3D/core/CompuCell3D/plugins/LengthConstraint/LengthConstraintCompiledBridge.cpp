#include "LengthConstraintPlugin.h"
#include <cc3d/CompiledSteppablePluginAPI.h>

namespace {
void set_length(void *instance, CC3DPluginCellHandle cell, double lambda, double target, double minor) {
    auto *plugin = static_cast<CompuCell3D::LengthConstraintPlugin *>(instance);
    plugin->setLengthConstraintData(static_cast<CompuCell3D::CellG *>(cell), lambda, target, minor);
}
double get_lambda(void *instance, CC3DPluginCellHandle cell) { return static_cast<CompuCell3D::LengthConstraintPlugin *>(instance)->getLambdaLength(static_cast<CompuCell3D::CellG *>(cell)); }
double get_target(void *instance, CC3DPluginCellHandle cell) { return static_cast<CompuCell3D::LengthConstraintPlugin *>(instance)->getTargetLength(static_cast<CompuCell3D::CellG *>(cell)); }
double get_minor_target(void *instance, CC3DPluginCellHandle cell) { return static_cast<CompuCell3D::LengthConstraintPlugin *>(instance)->getMinorTargetLength(static_cast<CompuCell3D::CellG *>(cell)); }
}

extern "C" void cc3d_register_length_constraint_plugin(void *simulator, void *plugin) {
    CC3DLengthConstraintCallbacksV1 callbacks{};
    callbacks.plugin_instance = plugin;
    callbacks.set_fn = &set_length;
    callbacks.get_lambda_fn = &get_lambda;
    callbacks.get_target_fn = &get_target;
    callbacks.get_minor_target_fn = &get_minor_target;
    cc3d_register_length_constraint_callbacks(simulator, &callbacks);
}
