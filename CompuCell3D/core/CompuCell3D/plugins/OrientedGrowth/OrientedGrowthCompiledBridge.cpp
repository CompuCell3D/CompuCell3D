#include "OrientedGrowthPlugin.h"
#include <cc3d/CompiledSteppablePluginAPI.h>

namespace {
void set_constraint_width(void *instance, CC3DPluginCellHandle cell, float value) {
    static_cast<CompuCell3D::OrientedGrowthPlugin *>(instance)->setConstraintWidth(
            static_cast<CompuCell3D::CellG *>(cell), value);
}
void set_elongation_axis(void *instance, CC3DPluginCellHandle cell, float x, float y) {
    static_cast<CompuCell3D::OrientedGrowthPlugin *>(instance)->setElongationAxis(
            static_cast<CompuCell3D::CellG *>(cell), x, y);
}
void set_elongation_enabled(void *instance, CC3DPluginCellHandle cell, uint8_t enabled) {
    static_cast<CompuCell3D::OrientedGrowthPlugin *>(instance)->setElongationEnabled(
            static_cast<CompuCell3D::CellG *>(cell), enabled != 0);
}
float get_elongation_axis_x(void *instance, CC3DPluginCellHandle cell) {
    return static_cast<CompuCell3D::OrientedGrowthPlugin *>(instance)->getElongationAxis_X(
            static_cast<CompuCell3D::CellG *>(cell));
}
float get_elongation_axis_y(void *instance, CC3DPluginCellHandle cell) {
    return static_cast<CompuCell3D::OrientedGrowthPlugin *>(instance)->getElongationAxis_Y(
            static_cast<CompuCell3D::CellG *>(cell));
}
uint8_t get_elongation_enabled(void *instance, CC3DPluginCellHandle cell) {
    return static_cast<CompuCell3D::OrientedGrowthPlugin *>(instance)->getElongationEnabled(
            static_cast<CompuCell3D::CellG *>(cell)) ? 1u : 0u;
}
}

extern "C" void cc3d_register_oriented_growth_plugin(void *simulator, void *plugin) {
    CC3DOrientedGrowthCallbacksV1 callbacks{};
    callbacks.plugin_instance = plugin;
    callbacks.set_constraint_width_fn = &set_constraint_width;
    callbacks.set_elongation_axis_fn = &set_elongation_axis;
    callbacks.set_elongation_enabled_fn = &set_elongation_enabled;
    callbacks.get_elongation_axis_x_fn = &get_elongation_axis_x;
    callbacks.get_elongation_axis_y_fn = &get_elongation_axis_y;
    callbacks.get_elongation_enabled_fn = &get_elongation_enabled;
    cc3d_register_oriented_growth_callbacks(simulator, &callbacks);
}
