#include "MitosisSteppable.h"
#include <cc3d/CompiledSteppablePluginAPI.h>

namespace {
CompuCell3D::MitosisSteppable *mitosis(void *instance) {
    return static_cast<CompuCell3D::MitosisSteppable *>(instance);
}
uint8_t divide_random(void *i, CC3DPluginCellHandle c) { return mitosis(i)->doDirectionalMitosisRandomOrientation(static_cast<CompuCell3D::CellG *>(c)); }
uint8_t divide_orientation(void *i, CC3DPluginCellHandle c, double x, double y, double z) { return mitosis(i)->doDirectionalMitosisOrientationVectorBased(static_cast<CompuCell3D::CellG *>(c), x, y, z); }
uint8_t divide_major(void *i, CC3DPluginCellHandle c) { return mitosis(i)->doDirectionalMitosisAlongMajorAxis(static_cast<CompuCell3D::CellG *>(c)); }
uint8_t divide_minor(void *i, CC3DPluginCellHandle c) { return mitosis(i)->doDirectionalMitosisAlongMinorAxis(static_cast<CompuCell3D::CellG *>(c)); }
uint8_t divide_cluster_random(void *i, long id) { return mitosis(i)->doDirectionalMitosisRandomOrientationCompartments(id); }
uint8_t divide_cluster_orientation(void *i, long id, double x, double y, double z) { return mitosis(i)->doDirectionalMitosisOrientationVectorBasedCompartments(id, x, y, z); }
uint8_t divide_cluster_major(void *i, long id) { return mitosis(i)->doDirectionalMitosisAlongMajorAxisCompartments(id); }
uint8_t divide_cluster_minor(void *i, long id) { return mitosis(i)->doDirectionalMitosisAlongMinorAxisCompartments(id); }
}

extern "C" void cc3d_register_mitosis_plugin(void *simulator, void *instance) {
    CC3DMitosisCallbacksV1 callbacks{};
    callbacks.steppable_instance = instance;
    callbacks.set_parent_child_position_fn = [](void *i, int v) { mitosis(i)->setParentChildPositionFlag(v); };
    callbacks.get_parent_child_position_fn = [](void *i) { return mitosis(i)->getParentChildPositionFlag(); };
    callbacks.divide_random_fn = &divide_random;
    callbacks.divide_orientation_fn = &divide_orientation;
    callbacks.divide_major_axis_fn = &divide_major;
    callbacks.divide_minor_axis_fn = &divide_minor;
    callbacks.divide_cluster_random_fn = &divide_cluster_random;
    callbacks.divide_cluster_orientation_fn = &divide_cluster_orientation;
    callbacks.divide_cluster_major_axis_fn = &divide_cluster_major;
    callbacks.divide_cluster_minor_axis_fn = &divide_cluster_minor;
    callbacks.parent_cell_fn = [](void *i) { return static_cast<CC3DPluginCellHandle>(mitosis(i)->parentCell); };
    callbacks.child_cell_fn = [](void *i) { return static_cast<CC3DPluginCellHandle>(mitosis(i)->childCell); };
    cc3d_register_mitosis_callbacks(simulator, &callbacks);
}
