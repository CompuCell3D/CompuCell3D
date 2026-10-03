#include "CompiledSteppable.h"
#include "cc3d/CompiledSteppablePluginAPI.h"

#include <algorithm>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <Python.h>

#include "CC3DExceptions.h"
#include "Automaton/Automaton.h"
#include "Field3D/Field3D.h"
#include "Field3D/NumpyArrayWrapper3DImpl.h"
#include "Field3D/VectorNumpyArrayWrapper3DImpl.h"
#include "PluginManager.h"
#include "plugins/AdhesionFlex/AdhesionFlexPlugin.h"
#include "plugins/BoundaryPixelTracker/BoundaryPixelTrackerPlugin.h"
#include "plugins/PixelTracker/PixelTrackerPlugin.h"
#include "plugins/NeighborTracker/NeighborTrackerPlugin.h"
#include "plugins/FocalPointPlasticity/FocalPointPlasticityPlugin.h"
#include "plugins/VolumeTracker/VolumeTrackerPlugin.h"
#include "plugins/ClusterSurfaceTracker/ClusterSurfaceTrackerPlugin.h"
#include "Potts3D/Cell.h"
#include "Potts3D/CellInventory.h"
#include "Potts3D/Potts3D.h"
#include "Simulator.h"
#include "XMLUtils/CC3DXMLElement.h"
// #include <Utils/Coordinates3D.h>

using namespace CompuCell3D;

namespace {
    struct LengthConstraintRegistration { void *simulator; CC3DLengthConstraintCallbacksV1 callbacks; };
    struct ClusterSurfaceRegistration { void *simulator; CC3DClusterSurfaceCallbacksV1 callbacks; };
    struct ClusterSurfaceTrackerRegistration { void *simulator; CC3DClusterSurfaceTrackerCallbacksV1 callbacks; };
    struct MitosisRegistration { void *simulator; CC3DMitosisCallbacksV1 callbacks; };
    struct OrientedGrowthRegistration { void *simulator; CC3DOrientedGrowthCallbacksV1 callbacks; };
    struct MomentOfInertiaRegistration { void *simulator; CC3DMomentOfInertiaCallbacksV1 callbacks; };
    struct AdhesionFlexRegistration { void *simulator; CC3DAdhesionFlexCallbacksV1 callbacks; };
    struct FocalPointPlasticityRegistration { void *simulator; CC3DFocalPointPlasticityCallbacksV1 callbacks; };
    struct PolarizationVectorRegistration { void *simulator; CC3DPolarizationVectorCallbacksV1 callbacks; };
    struct CellOrientationRegistration { void *simulator; CC3DCellOrientationCallbacksV1 callbacks; };
    struct ChemotaxisRegistration { void *simulator; CC3DChemotaxisCallbacksV1 callbacks; };
    struct SecretionRegistration { void *simulator; CC3DSecretionCallbacksV1 callbacks; };
    LengthConstraintRegistration lengthConstraintRegistration{};
    ClusterSurfaceRegistration clusterSurfaceRegistration{};
    ClusterSurfaceTrackerRegistration clusterSurfaceTrackerRegistration{};
    MitosisRegistration mitosisRegistration{};
    OrientedGrowthRegistration orientedGrowthRegistration{};
    MomentOfInertiaRegistration momentOfInertiaRegistration{};
    AdhesionFlexRegistration adhesionFlexRegistration{};
    FocalPointPlasticityRegistration focalPointPlasticityRegistration{};
    PolarizationVectorRegistration polarizationVectorRegistration{};
    CellOrientationRegistration cellOrientationRegistration{};
    ChemotaxisRegistration chemotaxisRegistration{};
    SecretionRegistration secretionRegistration{};

    struct AdhesionFlexBridge {
        const CC3DAdhesionFlexCallbacksV1 *callbacks;
        std::vector<std::string> names;
        size_t getAdhesionMoleculeNameVecSize() { return callbacks && callbacks->molecule_count_fn ? callbacks->molecule_count_fn(callbacks->plugin_instance) : 0u; }
        std::vector<std::string> getAdhesionMoleculeNameVec() { names.clear(); for (size_t i = 0; i < getAdhesionMoleculeNameVecSize(); ++i) { const char *name = callbacks->molecule_name_fn(callbacks->plugin_instance, i); if (name) names.emplace_back(name); } return names; }
        float getAdhesionMoleculeDensity(CompuCell3D::CellG *c, const char *n) { return callbacks->get_density_fn(callbacks->plugin_instance, c, n); }
        float getAdhesionMoleculeDensityByIndex(CompuCell3D::CellG *c, int i) { return callbacks->get_density_by_index_fn(callbacks->plugin_instance, c, static_cast<size_t>(i)); }
        void setAdhesionMoleculeDensity(CompuCell3D::CellG *c, const char *n, float v) { callbacks->set_density_fn(callbacks->plugin_instance, c, n, v); }
        void setAdhesionMoleculeDensityByIndex(CompuCell3D::CellG *c, int i, float v) { callbacks->set_density_by_index_fn(callbacks->plugin_instance, c, static_cast<size_t>(i), v); }
        void setAdhesionMoleculeDensityVector(CompuCell3D::CellG *c, std::vector<float> v) { callbacks->set_density_vector_fn(callbacks->plugin_instance, c, v.data(), v.size()); }
        void assignNewAdhesionMoleculeDensityVector(CompuCell3D::CellG *c, std::vector<float> v) { callbacks->assign_density_vector_fn(callbacks->plugin_instance, c, v.data(), v.size()); }
        float getMediumAdhesionMoleculeDensity(const char *n) { return callbacks->get_medium_density_fn(callbacks->plugin_instance, n); }
        float getMediumAdhesionMoleculeDensityByIndex(int i) { return callbacks->get_medium_density_by_index_fn(callbacks->plugin_instance, static_cast<size_t>(i)); }
        void setMediumAdhesionMoleculeDensity(const char *n, float v) { callbacks->set_medium_density_fn(callbacks->plugin_instance, n, v); }
        void setMediumAdhesionMoleculeDensityByIndex(int i, float v) { callbacks->set_medium_density_by_index_fn(callbacks->plugin_instance, static_cast<size_t>(i), v); }
        void setMediumAdhesionMoleculeDensityVector(std::vector<float> v) { callbacks->set_medium_density_vector_fn(callbacks->plugin_instance, v.data(), v.size()); }
        void assignNewMediumAdhesionMoleculeDensityVector(std::vector<float> v) { callbacks->assign_medium_density_vector_fn(callbacks->plugin_instance, v.data(), v.size()); }
    };
    struct FocalPointPlasticityBridge {
        const CC3DFocalPointPlasticityCallbacksV1 *callbacks;
        void createFocalPointPlasticityLink(CompuCell3D::CellG *a, CompuCell3D::CellG *b, double l, double t, double m) { callbacks->create_link_fn(callbacks->plugin_instance, a, b, l, t, m); }
        void createInternalFocalPointPlasticityLink(CompuCell3D::CellG *a, CompuCell3D::CellG *b, double l, double t, double m) { callbacks->create_internal_link_fn(callbacks->plugin_instance, a, b, l, t, m); }
        void deleteFocalPointPlasticityLink(CompuCell3D::CellG *a, CompuCell3D::CellG *b) { callbacks->delete_link_fn(callbacks->plugin_instance, a, b); }
        void deleteInternalFocalPointPlasticityLink(CompuCell3D::CellG *a, CompuCell3D::CellG *b) { callbacks->delete_internal_link_fn(callbacks->plugin_instance, a, b); }
        void setFocalPointPlasticityParameters(CompuCell3D::CellG *a, CompuCell3D::CellG *b, double l, double t, double m) { callbacks->set_link_fn(callbacks->plugin_instance, a, b, l, t, m); }
        double getPlasticityParametersLambdaDistance(CompuCell3D::CellG *a, CompuCell3D::CellG *b) { return callbacks->get_lambda_fn(callbacks->plugin_instance, a, b); }
        double getPlasticityParametersTargetDistance(CompuCell3D::CellG *a, CompuCell3D::CellG *b) { return callbacks->get_target_fn(callbacks->plugin_instance, a, b); }
        int createAnchor(CompuCell3D::CellG *c, double l, double t, double m, float x, float y, float z) { return callbacks->create_anchor_fn(callbacks->plugin_instance, c, l, t, m, x, y, z); }
        void deleteAnchor(CompuCell3D::CellG *c, int id) { callbacks->delete_anchor_fn(callbacks->plugin_instance, c, id); }
        void setAnchorParameters(CompuCell3D::CellG *c, int id, double l, double t, double m, float x, float y, float z) { callbacks->set_anchor_fn(callbacks->plugin_instance, c, id, l, t, m, x, y, z); }
    };
    AdhesionFlexBridge *cc3d_ensure_adhesion_flex(void *);
    const CC3DLengthConstraintCallbacksV1 *cc3d_ensure_length_constraint(void *);
    const CC3DPolarizationVectorCallbacksV1 *cc3d_ensure_polarization_vector(void *);
    const CC3DCellOrientationCallbacksV1 *cc3d_ensure_cell_orientation(void *);
    const CC3DChemotaxisCallbacksV1 *cc3d_ensure_chemotaxis(void *);
}

extern "C" void cc3d_register_length_constraint_callbacks(void *simulator, const CC3DLengthConstraintCallbacksV1 *callbacks) {
    if (!simulator || !callbacks) return;
    lengthConstraintRegistration = {simulator, *callbacks};
}

extern "C" const CC3DLengthConstraintCallbacksV1 *cc3d_get_length_constraint_callbacks(void *simulator) {
    return lengthConstraintRegistration.simulator == simulator ? &lengthConstraintRegistration.callbacks : nullptr;
}

extern "C" void cc3d_unregister_length_constraint_callbacks(void *plugin_instance) {
    if (lengthConstraintRegistration.callbacks.plugin_instance == plugin_instance) lengthConstraintRegistration = {};
}

extern "C" void cc3d_register_cluster_surface_callbacks(void *simulator, const CC3DClusterSurfaceCallbacksV1 *callbacks) {
    if (!simulator || !callbacks) return;
    clusterSurfaceRegistration = {simulator, *callbacks};
}
extern "C" const CC3DClusterSurfaceCallbacksV1 *cc3d_get_cluster_surface_callbacks(void *simulator) {
    return clusterSurfaceRegistration.simulator == simulator ? &clusterSurfaceRegistration.callbacks : nullptr;
}
extern "C" void cc3d_unregister_cluster_surface_callbacks(void *plugin_instance) {
    if (clusterSurfaceRegistration.callbacks.plugin_instance == plugin_instance) clusterSurfaceRegistration = {};
}

extern "C" void cc3d_register_cluster_surface_tracker_callbacks(void *simulator, const CC3DClusterSurfaceTrackerCallbacksV1 *callbacks) {
    if (!simulator || !callbacks) return;
    clusterSurfaceTrackerRegistration = {simulator, *callbacks};
}
extern "C" const CC3DClusterSurfaceTrackerCallbacksV1 *cc3d_get_cluster_surface_tracker_callbacks(void *simulator) {
    return clusterSurfaceTrackerRegistration.simulator == simulator ? &clusterSurfaceTrackerRegistration.callbacks : nullptr;
}
extern "C" void cc3d_unregister_cluster_surface_tracker_callbacks(void *plugin_instance) {
    if (clusterSurfaceTrackerRegistration.callbacks.plugin_instance == plugin_instance) clusterSurfaceTrackerRegistration = {};
}

extern "C" void cc3d_register_mitosis_callbacks(void *simulator, const CC3DMitosisCallbacksV1 *callbacks) {
    if (!simulator || !callbacks) return;
    mitosisRegistration = {simulator, *callbacks};
}
extern "C" const CC3DMitosisCallbacksV1 *cc3d_get_mitosis_callbacks(void *simulator) {
    return mitosisRegistration.simulator == simulator ? &mitosisRegistration.callbacks : nullptr;
}
extern "C" void cc3d_unregister_mitosis_callbacks(void *steppable_instance) {
    if (mitosisRegistration.callbacks.steppable_instance == steppable_instance) mitosisRegistration = {};
}

extern "C" void cc3d_register_oriented_growth_callbacks(void *simulator, const CC3DOrientedGrowthCallbacksV1 *callbacks) {
    if (!simulator || !callbacks) return;
    orientedGrowthRegistration = {simulator, *callbacks};
}
extern "C" const CC3DOrientedGrowthCallbacksV1 *cc3d_get_oriented_growth_callbacks(void *simulator) {
    return orientedGrowthRegistration.simulator == simulator ? &orientedGrowthRegistration.callbacks : nullptr;
}
extern "C" void cc3d_unregister_oriented_growth_callbacks(void *plugin_instance) {
    if (orientedGrowthRegistration.callbacks.plugin_instance == plugin_instance) orientedGrowthRegistration = {};
}

extern "C" void cc3d_register_moment_of_inertia_callbacks(void *simulator, const CC3DMomentOfInertiaCallbacksV1 *callbacks) {
    if (!simulator || !callbacks) return;
    momentOfInertiaRegistration = {simulator, *callbacks};
}
extern "C" const CC3DMomentOfInertiaCallbacksV1 *cc3d_get_moment_of_inertia_callbacks(void *simulator) {
    return momentOfInertiaRegistration.simulator == simulator ? &momentOfInertiaRegistration.callbacks : nullptr;
}
extern "C" void cc3d_unregister_moment_of_inertia_callbacks(void *plugin_instance) {
    if (momentOfInertiaRegistration.callbacks.plugin_instance == plugin_instance) momentOfInertiaRegistration = {};
}

extern "C" void cc3d_register_adhesion_flex_callbacks(void *simulator, const CC3DAdhesionFlexCallbacksV1 *callbacks) {
    if (!simulator || !callbacks) return;
    adhesionFlexRegistration = {simulator, *callbacks};
}
extern "C" const CC3DAdhesionFlexCallbacksV1 *cc3d_get_adhesion_flex_callbacks(void *simulator) {
    return adhesionFlexRegistration.simulator == simulator ? &adhesionFlexRegistration.callbacks : nullptr;
}
extern "C" void cc3d_unregister_adhesion_flex_callbacks(void *plugin_instance) {
    if (adhesionFlexRegistration.callbacks.plugin_instance == plugin_instance) adhesionFlexRegistration = {};
}
extern "C" void cc3d_register_focal_point_plasticity_callbacks(void *simulator, const CC3DFocalPointPlasticityCallbacksV1 *callbacks) {
    if (!simulator || !callbacks) return;
    focalPointPlasticityRegistration = {simulator, *callbacks};
}
extern "C" const CC3DFocalPointPlasticityCallbacksV1 *cc3d_get_focal_point_plasticity_callbacks(void *simulator) {
    return focalPointPlasticityRegistration.simulator == simulator ? &focalPointPlasticityRegistration.callbacks : nullptr;
}
extern "C" void cc3d_unregister_focal_point_plasticity_callbacks(void *plugin_instance) {
    if (focalPointPlasticityRegistration.callbacks.plugin_instance == plugin_instance) focalPointPlasticityRegistration = {};
}
extern "C" void cc3d_register_polarization_vector_callbacks(void *simulator, const CC3DPolarizationVectorCallbacksV1 *callbacks) {
    if (!simulator || !callbacks) return;
    polarizationVectorRegistration = {simulator, *callbacks};
}
extern "C" const CC3DPolarizationVectorCallbacksV1 *cc3d_get_polarization_vector_callbacks(void *simulator) {
    return polarizationVectorRegistration.simulator == simulator ? &polarizationVectorRegistration.callbacks : nullptr;
}
extern "C" void cc3d_unregister_polarization_vector_callbacks(void *plugin_instance) {
    if (polarizationVectorRegistration.callbacks.plugin_instance == plugin_instance) polarizationVectorRegistration = {};
}
extern "C" void cc3d_register_cell_orientation_callbacks(void *simulator, const CC3DCellOrientationCallbacksV1 *callbacks) {
    if (!simulator || !callbacks) return;
    cellOrientationRegistration = {simulator, *callbacks};
}
extern "C" const CC3DCellOrientationCallbacksV1 *cc3d_get_cell_orientation_callbacks(void *simulator) {
    return cellOrientationRegistration.simulator == simulator ? &cellOrientationRegistration.callbacks : nullptr;
}
extern "C" void cc3d_unregister_cell_orientation_callbacks(void *plugin_instance) {
    if (cellOrientationRegistration.callbacks.plugin_instance == plugin_instance) cellOrientationRegistration = {};
}
extern "C" void cc3d_register_chemotaxis_callbacks(void *simulator, const CC3DChemotaxisCallbacksV1 *callbacks) {
    if (!simulator || !callbacks) return;
    chemotaxisRegistration = {simulator, *callbacks};
}
extern "C" const CC3DChemotaxisCallbacksV1 *cc3d_get_chemotaxis_callbacks(void *simulator) {
    return chemotaxisRegistration.simulator == simulator ? &chemotaxisRegistration.callbacks : nullptr;
}
extern "C" void cc3d_unregister_chemotaxis_callbacks(void *plugin_instance) {
    if (chemotaxisRegistration.callbacks.plugin_instance == plugin_instance) chemotaxisRegistration = {};
}
extern "C" void cc3d_register_secretion_callbacks(void *simulator, const CC3DSecretionCallbacksV1 *callbacks) {
    if (!simulator || !callbacks) return;
    secretionRegistration = {simulator, *callbacks};
}
extern "C" const CC3DSecretionCallbacksV1 *cc3d_get_secretion_callbacks(void *simulator) {
    return secretionRegistration.simulator == simulator ? &secretionRegistration.callbacks : nullptr;
}
extern "C" void cc3d_unregister_secretion_callbacks(void *plugin_instance) {
    if (secretionRegistration.callbacks.plugin_instance == plugin_instance) secretionRegistration = {};
}

CC3DCellViewV1 cc3d_cell_view_from_cell(CompuCell3D::CellG *cell) {
        CC3DCellViewV1 view{};

        if (!cell) {
            return view;
        }

        view.handle = static_cast<CC3DCellHandle>(cell);
        view.id = &cell->id;
        view.clusterId = &cell->clusterId;
        view.volume = &cell->volume;
        view.surface = &cell->surface;
        view.clusterSurface = &cell->clusterSurface;
        view.type = &cell->type;
        view.subtype = &cell->subtype;
        view.targetVolume = &cell->targetVolume;
        view.lambdaVolume = &cell->lambdaVolume;
        view.targetSurface = &cell->targetSurface;
        view.angle = &cell->angle;
        view.lambdaSurface = &cell->lambdaSurface;
        view.targetClusterSurface = &cell->targetClusterSurface;
        view.lambdaClusterSurface = &cell->lambdaClusterSurface;
        view.xCM = &cell->xCM;
        view.yCM = &cell->yCM;
        view.zCM = &cell->zCM;
        view.xCOM = &cell->xCOM;
        view.yCOM = &cell->yCOM;
        view.zCOM = &cell->zCOM;
        view.xCOMPrev = &cell->xCOMPrev;
        view.yCOMPrev = &cell->yCOMPrev;
        view.zCOMPrev = &cell->zCOMPrev;
        view.iXX = &cell->iXX;
        view.iXY = &cell->iXY;
        view.iXZ = &cell->iXZ;
        view.iYY = &cell->iYY;
        view.iYZ = &cell->iYZ;
        view.iZZ = &cell->iZZ;
        view.lX = &cell->lX;
        view.lY = &cell->lY;
        view.lZ = &cell->lZ;
        view.ecc = &cell->ecc;
        view.lambdaVecX = &cell->lambdaVecX;
        view.lambdaVecY = &cell->lambdaVecY;
        view.lambdaVecZ = &cell->lambdaVecZ;
        view.flag = &cell->flag;
        view.averageConcentration = &cell->averageConcentration;
        view.fluctAmpl = &cell->fluctAmpl;
        view.lambdaMotility = &cell->lambdaMotility;
        view.biasVecX = &cell->biasVecX;
        view.biasVecY = &cell->biasVecY;
        view.biasVecZ = &cell->biasVecZ;
        view.connectivityOn = &cell->connectivityOn;
        return view;
}

namespace {
    struct XmlElementHandle {
        Simulator *simulator;
        std::string elementId;
    };

    CC3DXMLElement *cc3d_find_xml_element(CC3DXMLElement *root, const std::string &elementId) {
        if (!root) return nullptr;
        if (root->findAttribute("id") && root->getAttribute("id") == elementId) return root;
        for (auto *child : root->children) {
            if (auto *found = cc3d_find_xml_element(child, elementId)) return found;
        }
        return nullptr;
    }

    CC3DXMLElement *cc3d_find_xml_owner(
        Simulator *simulator, const std::string &elementId, CC3DXMLElement *&element
    ) {
        if (!simulator) return nullptr;
        std::vector<CC3DXMLElement *> roots;
        roots.push_back(simulator->ps.pottsCC3DXMLElement);
        roots.push_back(simulator->ps.metadataCC3DXMLElement);
        for (auto *root : simulator->ps.pluginCC3DXMLElementVector) roots.push_back(root);
        for (auto *root : simulator->ps.steppableCC3DXMLElementVector) roots.push_back(root);
        for (auto *root : roots) {
            element = cc3d_find_xml_element(root, elementId);
            if (element) return root;
        }
        return nullptr;
    }

    CC3DXMLHandle cc3d_xml_get_element(void *simulatorPtr, const char *elementId) {
        if (!simulatorPtr || !elementId) return nullptr;
        auto *simulator = static_cast<Simulator *>(simulatorPtr);
        CC3DXMLElement *element = nullptr;
        auto *owner = cc3d_find_xml_owner(simulator, elementId, element);
        if (!element || !owner) return nullptr;
        return static_cast<CC3DXMLHandle>(new XmlElementHandle{simulator, elementId});
    }

    void cc3d_xml_destroy_element(void *, CC3DXMLHandle handle) {
        delete static_cast<XmlElementHandle *>(handle);
    }

    bool cc3d_xml_is_cdata(const char *field) {
        return field && std::string(field) == "cdata";
    }

    CC3DXMLElement *cc3d_xml_resolve_element(
        XmlElementHandle *handle, CC3DXMLElement *&owner
    ) {
        if (!handle || !handle->simulator) return nullptr;
        CC3DXMLElement *element = nullptr;
        owner = cc3d_find_xml_owner(handle->simulator, handle->elementId, element);
        return owner && element ? element : nullptr;
    }

    bool cc3d_xml_get_string(XmlElementHandle *handle, const char *field, std::string &value) {
        if (!handle || !field) return false;
        try {
            CC3DXMLElement *owner = nullptr;
            auto *element = cc3d_xml_resolve_element(handle, owner);
            if (!element) return false;
            if (cc3d_xml_is_cdata(field)) {
                value = element->cdata;
            } else {
                if (!element->findAttribute(field)) return false;
                value = element->getAttribute(field);
            }
            return true;
        } catch (...) {
            return false;
        }
    }

    uint8_t cc3d_xml_get_value(
        void *, CC3DXMLHandle handle, const char *field, char *buffer, size_t capacity
    ) {
        std::string value;
        if (!cc3d_xml_get_string(static_cast<XmlElementHandle *>(handle), field, value) ||
            !buffer || capacity <= value.size()) return 0u;
        std::memcpy(buffer, value.c_str(), value.size() + 1u);
        return 1u;
    }

    uint8_t cc3d_xml_get_value_as_double(
        void *, CC3DXMLHandle handle, const char *field, double *value
    ) {
        if (!handle || !field || !value) return 0u;
        try {
            CC3DXMLElement *owner = nullptr;
            auto *xml = cc3d_xml_resolve_element(
                static_cast<XmlElementHandle *>(handle), owner
            );
            if (!xml) return 0u;
            *value = cc3d_xml_is_cdata(field)
                ? xml->getDouble()
                : xml->getAttributeAsDouble(field);
            return 1u;
        } catch (...) {
            return 0u;
        }
    }

    bool cc3d_xml_set_string(
        Simulator *simulator, XmlElementHandle *handle, const char *field, const char *value
    ) {
        if (!simulator || !handle || !field || !value ||
            std::string(field) == "id") return false;
        try {
            CC3DXMLElement *owner = nullptr;
            auto *element = cc3d_xml_resolve_element(handle, owner);
            if (!element || !owner) return false;
            if (cc3d_xml_is_cdata(field)) {
                element->updateElementValue(value);
            } else {
                if (!element->findAttribute(field)) return false;
                std::map<std::string, std::string> attributes;
                attributes[field] = value;
                element->updateElementAttributes(&attributes);
            }
            simulator->updateCC3DModule(owner);
            return true;
        } catch (...) {
            return false;
        }
    }

    uint8_t cc3d_xml_set_value(
        void *simulatorPtr, CC3DXMLHandle handle, const char *field, const char *value
    ) {
        return cc3d_xml_set_string(
            static_cast<Simulator *>(simulatorPtr),
            static_cast<XmlElementHandle *>(handle), field, value
        ) ? 1u : 0u;
    }

    uint8_t cc3d_xml_set_value_as_double(
        void *simulatorPtr, CC3DXMLHandle handle, const char *field, double value
    ) {
        std::ostringstream stream;
        stream << value;
        return cc3d_xml_set_value(simulatorPtr, handle, field, stream.str().c_str());
    }

    PyObject *cc3d_python_dictionary(void *userdata, CC3DCellHandle cellHandle, bool create) {
        if (cellHandle) {
            auto *cell = static_cast<CompuCell3D::CellG *>(cellHandle);
            if (!cell->pyAttrib && create) cell->pyAttrib = PyDict_New();
            return cell->pyAttrib;
        }
        return static_cast<PyObject *>(userdata);
    }

    bool cc3d_python_key_valid(const char *key) { return key && *key; }

    uint8_t cc3d_python_dict_contains(void *userdata, CC3DCellHandle cellHandle, const char *key) {
        PyGILState_STATE state = PyGILState_Ensure();
        PyObject *dict = cc3d_python_dictionary(userdata, cellHandle, false);
        const int result = dict && cc3d_python_key_valid(key) && PyDict_GetItemString(dict, key) != nullptr;
        PyGILState_Release(state);
        return result ? 1u : 0u;
    }

    void cc3d_python_dict_erase(void *userdata, CC3DCellHandle cellHandle, const char *key) {
        PyGILState_STATE state = PyGILState_Ensure();
        PyObject *dict = cc3d_python_dictionary(userdata, cellHandle, false);
        if (dict && cc3d_python_key_valid(key)) PyDict_DelItemString(dict, key);
        PyErr_Clear();
        PyGILState_Release(state);
    }

    uint8_t cc3d_python_dict_set_bool(void *userdata, CC3DCellHandle cellHandle, const char *key, uint8_t value) {
        PyGILState_STATE state = PyGILState_Ensure();
        PyObject *dict = cc3d_python_dictionary(userdata, cellHandle, true);
        PyObject *item = PyBool_FromLong(value != 0);
        const int result = dict && item && cc3d_python_key_valid(key) && PyDict_SetItemString(dict, key, item) == 0;
        Py_XDECREF(item);
        PyGILState_Release(state);
        return result ? 1u : 0u;
    }

    uint8_t cc3d_python_dict_set_int(void *userdata, CC3DCellHandle cellHandle, const char *key, int64_t value) {
        PyGILState_STATE state = PyGILState_Ensure();
        PyObject *dict = cc3d_python_dictionary(userdata, cellHandle, true);
        PyObject *item = PyLong_FromLongLong(value);
        const int result = dict && item && cc3d_python_key_valid(key) && PyDict_SetItemString(dict, key, item) == 0;
        Py_XDECREF(item);
        PyGILState_Release(state);
        return result ? 1u : 0u;
    }

    uint8_t cc3d_python_dict_set_double(void *userdata, CC3DCellHandle cellHandle, const char *key, double value) {
        PyGILState_STATE state = PyGILState_Ensure();
        PyObject *dict = cc3d_python_dictionary(userdata, cellHandle, true);
        PyObject *item = PyFloat_FromDouble(value);
        const int result = dict && item && cc3d_python_key_valid(key) && PyDict_SetItemString(dict, key, item) == 0;
        Py_XDECREF(item);
        PyGILState_Release(state);
        return result ? 1u : 0u;
    }

    uint8_t cc3d_python_dict_set_string(void *userdata, CC3DCellHandle cellHandle, const char *key, const char *value) {
        PyGILState_STATE state = PyGILState_Ensure();
        PyObject *dict = cc3d_python_dictionary(userdata, cellHandle, true);
        PyObject *item = value ? PyUnicode_FromString(value) : nullptr;
        const int result = dict && item && cc3d_python_key_valid(key) && PyDict_SetItemString(dict, key, item) == 0;
        Py_XDECREF(item);
        PyGILState_Release(state);
        return result ? 1u : 0u;
    }

    uint8_t cc3d_python_dict_set_tuple(void *userdata, CC3DCellHandle cellHandle, const char *key,
                                       const double *values, size_t count) {
        PyGILState_STATE state = PyGILState_Ensure();
        PyObject *dict = cc3d_python_dictionary(userdata, cellHandle, true);
        PyObject *item = PyTuple_New(count);
        if (item && values) {
            for (size_t i = 0; i < count; ++i) PyTuple_SET_ITEM(item, i, PyFloat_FromDouble(values[i]));
        }
        const int result = dict && item && cc3d_python_key_valid(key) && PyDict_SetItemString(dict, key, item) == 0;
        Py_XDECREF(item);
        PyGILState_Release(state);
        return result ? 1u : 0u;
    }

    uint8_t cc3d_python_dict_get_bool(void *userdata, CC3DCellHandle cellHandle, const char *key, uint8_t *value) {
        PyGILState_STATE state = PyGILState_Ensure();
        PyObject *item = cc3d_python_dictionary(userdata, cellHandle, false);
        item = item && cc3d_python_key_valid(key) ? PyDict_GetItemString(item, key) : nullptr;
        const int result = item && PyBool_Check(item) && value;
        if (result) *value = item == Py_True ? 1u : 0u;
        PyGILState_Release(state);
        return result ? 1u : 0u;
    }

    uint8_t cc3d_python_dict_get_int(void *userdata, CC3DCellHandle cellHandle, const char *key, int64_t *value) {
        PyGILState_STATE state = PyGILState_Ensure();
        PyObject *dict = cc3d_python_dictionary(userdata, cellHandle, false);
        PyObject *item = dict && cc3d_python_key_valid(key) ? PyDict_GetItemString(dict, key) : nullptr;
        const int result = item && PyLong_Check(item) && !PyBool_Check(item) && value;
        if (result) *value = PyLong_AsLongLong(item);
        PyErr_Clear();
        PyGILState_Release(state);
        return result ? 1u : 0u;
    }

    uint8_t cc3d_python_dict_get_double(void *userdata, CC3DCellHandle cellHandle, const char *key, double *value) {
        PyGILState_STATE state = PyGILState_Ensure();
        PyObject *dict = cc3d_python_dictionary(userdata, cellHandle, false);
        PyObject *item = dict && cc3d_python_key_valid(key) ? PyDict_GetItemString(dict, key) : nullptr;
        const int result = item && (PyFloat_Check(item) || (PyLong_Check(item) && !PyBool_Check(item))) && value;
        if (result) *value = PyFloat_AsDouble(item);
        PyErr_Clear();
        PyGILState_Release(state);
        return result ? 1u : 0u;
    }

    size_t cc3d_python_dict_get_string(void *userdata, CC3DCellHandle cellHandle, const char *key,
                                       char *buffer, size_t capacity) {
        PyGILState_STATE state = PyGILState_Ensure();
        PyObject *dict = cc3d_python_dictionary(userdata, cellHandle, false);
        PyObject *item = dict && cc3d_python_key_valid(key) ? PyDict_GetItemString(dict, key) : nullptr;
        const char *value = item && PyUnicode_Check(item) ? PyUnicode_AsUTF8(item) : nullptr;
        const size_t size = value ? strlen(value) : 0u;
        if (value && buffer && capacity) {
            const size_t copySize = size < capacity - 1 ? size : capacity - 1;
            memcpy(buffer, value, copySize);
            buffer[copySize] = '\0';
        }
        PyErr_Clear();
        PyGILState_Release(state);
        return value ? size : 0u;
    }

    size_t cc3d_python_dict_get_tuple(void *userdata, CC3DCellHandle cellHandle, const char *key,
                                      double *buffer, size_t capacity) {
        PyGILState_STATE state = PyGILState_Ensure();
        PyObject *dict = cc3d_python_dictionary(userdata, cellHandle, false);
        PyObject *item = dict && cc3d_python_key_valid(key) ? PyDict_GetItemString(dict, key) : nullptr;
        const bool valid = item && PyTuple_Check(item);
        const size_t size = valid ? PyTuple_Size(item) : 0u;
        if (valid && buffer) {
            const size_t copySize = size < capacity ? size : capacity;
            for (size_t i = 0; i < copySize; ++i) buffer[i] = PyFloat_AsDouble(PyTuple_GetItem(item, i));
        }
        PyErr_Clear();
        PyGILState_Release(state);
        return valid ? size : 0u;
    }

    void cc3d_initialize_python_dictionary_api(CC3DPythonDictionaryAPIV1 &api, void *userdata) {
        api.userdata = userdata;
        api.contains_fn = &cc3d_python_dict_contains;
        api.erase_fn = &cc3d_python_dict_erase;
        api.set_bool_fn = &cc3d_python_dict_set_bool;
        api.set_int_fn = &cc3d_python_dict_set_int;
        api.set_double_fn = &cc3d_python_dict_set_double;
        api.set_string_fn = &cc3d_python_dict_set_string;
        api.set_tuple_fn = &cc3d_python_dict_set_tuple;
        api.get_bool_fn = &cc3d_python_dict_get_bool;
        api.get_int_fn = &cc3d_python_dict_get_int;
        api.get_double_fn = &cc3d_python_dict_get_double;
        api.get_string_fn = &cc3d_python_dict_get_string;
        api.get_tuple_fn = &cc3d_python_dict_get_tuple;
    }

    PyObject *cc3d_acquire_shared_steppable_vars() {
        PyGILState_STATE state = PyGILState_Ensure();
        // The dictionary is owned by the PersistentGlobals singleton exposed
        // from cc3d.CompuCellSetup, not by the persistent_globals module.
        PyObject *module = PyImport_ImportModule("cc3d.CompuCellSetup");
        PyObject *globals = module ? PyObject_GetAttrString(module, "persistent_globals") : nullptr;
        PyObject *shared = globals ? PyObject_GetAttrString(globals, "shared_steppable_vars") : nullptr;
        Py_XDECREF(globals);
        Py_XDECREF(module);
        if (!shared || !PyDict_Check(shared)) {
            Py_XDECREF(shared);
            shared = PyDict_New();
        }
        // Do not let an intermediate import/attribute lookup failure escape
        // through the Python/C++ boundary when the fallback is usable.
        PyErr_Clear();
        PyGILState_Release(state);
        return shared;
    }

    void cc3d_release_python_object(void *object) {
        if (!object) return;
        PyGILState_STATE state = PyGILState_Ensure();
        Py_DECREF(static_cast<PyObject *>(object));
        PyGILState_Release(state);
    }

    struct CellIteratorState {
        CompuCell3D::CellInventory::cellInventoryIterator current;
        CompuCell3D::CellInventory::cellInventoryIterator end;
        std::vector<unsigned char> typeIds;
    };

    struct CellNeighborIteratorState {
        std::set<CompuCell3D::NeighborSurfaceData>::iterator current;
        std::set<CompuCell3D::NeighborSurfaceData>::iterator end;
    };


    struct PixelIteratorState {
        std::set<CompuCell3D::PixelTrackerData>::iterator current;
        std::set<CompuCell3D::PixelTrackerData>::iterator end;
    };

    struct BoundaryPixelIteratorState {
        std::set<CompuCell3D::BoundaryPixelTrackerData>::iterator current;
        std::set<CompuCell3D::BoundaryPixelTrackerData>::iterator end;
    };

    CC3DCellIteratorHandle cc3d_cells_begin(void *cellInventoryPtr) {
        auto *inventory = static_cast<CompuCell3D::CellInventory *>(cellInventoryPtr);
        auto *state = new CellIteratorState{inventory->cellInventoryBegin(), inventory->cellInventoryEnd()};
        return static_cast<CC3DCellIteratorHandle>(state);
    }

    bool cc3d_cell_type_matches(const CellIteratorState *state) {
        if (!state || state->current == state->end) return false;
        if (state->typeIds.empty()) return true;

        const unsigned char cellType = state->current->second->type;
        return std::find(state->typeIds.begin(), state->typeIds.end(), cellType) != state->typeIds.end();
    }

    void cc3d_cells_advance_to_matching_type(CellIteratorState *state) {
        while (state && state->current != state->end && !cc3d_cell_type_matches(state)) {
            ++state->current;
        }
    }

    CC3DCellIteratorHandle cc3d_cells_by_type_begin(
        void *cellInventoryPtr, const unsigned char *typeIds, size_t typeCount
    ) {
        auto *inventory = static_cast<CompuCell3D::CellInventory *>(cellInventoryPtr);
        std::vector<unsigned char> requestedTypes;
        if (typeIds && typeCount) requestedTypes.assign(typeIds, typeIds + typeCount);

        auto *state = new CellIteratorState{
            inventory->cellInventoryBegin(), inventory->cellInventoryEnd(), std::move(requestedTypes)
        };
        cc3d_cells_advance_to_matching_type(state);
        return static_cast<CC3DCellIteratorHandle>(state);
    }

    CC3DCellViewV1 cc3d_cells_fetch_by_id(void *cellInventoryPtr, long cellId) {
        auto *inventory = static_cast<CompuCell3D::CellInventory *>(cellInventoryPtr);
        if (!inventory || inventory->getCellInventorySize() == 0u) return CC3DCellViewV1{};
        auto *cell = inventory->attemptFetchingCellById(cellId);
        return cell ? cc3d_cell_view_from_cell(cell) : CC3DCellViewV1{};
    }

    uint8_t cc3d_cells_valid(void *, CC3DCellIteratorHandle iterator) {
        auto *state = static_cast<CellIteratorState *>(iterator);
        return cc3d_cell_type_matches(state) ? 1u : 0u;
    }

    Point3D cc3d_point3d(int x, int y, int z) {
        return Point3D(static_cast<short>(x), static_cast<short>(y), static_cast<short>(z));
    }

    CompuCell3D::Simulator *cc3d_simulator_from_ptr(void *simulatorPtr) {
        return static_cast<CompuCell3D::Simulator *>(simulatorPtr);
    }

    CompuCell3D::Potts3D *cc3d_potts_from_simulator(void *simulatorPtr) {
        auto *simulator = cc3d_simulator_from_ptr(simulatorPtr);
        return simulator ? simulator->getPotts() : nullptr;
    }

    uint8_t cc3d_cell_types_get(void *simulatorPtr, const char *typeName, unsigned char *typeId) {
        if (!typeName || !typeId) return 0u;

        auto *potts = cc3d_potts_from_simulator(simulatorPtr);
        auto *automaton = potts ? potts->getAutomaton() : nullptr;
        if (!automaton) return 0u;

        try {
            *typeId = automaton->getTypeId(typeName);
            return 1u;
        } catch (...) {
            return 0u;
        }
    }

    size_t cc3d_cell_types_name(
        void *simulatorPtr, unsigned char typeId, char *buffer, size_t capacity
    ) {
        auto *potts = cc3d_potts_from_simulator(simulatorPtr);
        auto *automaton = potts ? potts->getAutomaton() : nullptr;
        if (!automaton) return 0u;

        std::string typeName;
        try {
            typeName = automaton->getTypeName(static_cast<char>(typeId));
        } catch (...) {
            return 0u;
        }

        const size_t size = typeName.size();
        if (buffer && capacity) {
            const size_t copySize = size < capacity - 1u ? size : capacity - 1u;
            std::memcpy(buffer, typeName.data(), copySize);
            buffer[copySize] = '\0';
        }
        return size;
    }

    CompuCell3D::Field3D<CompuCell3D::CellG *> *cc3d_cell_field_from_simulator(void *simulatorPtr) {
        auto *potts = cc3d_potts_from_simulator(simulatorPtr);
        return potts ? potts->getCellFieldG() : nullptr;
    }

    CompuCell3D::VolumeTrackerPlugin *cc3d_ensure_volume_tracker(void *simulatorPtr) {
        auto *simulator = cc3d_simulator_from_ptr(simulatorPtr);
        if (!simulator) {
            return nullptr;
        }

        bool pluginAlreadyRegisteredFlag = false;
        auto *plugin = CompuCell3D::Simulator::pluginManager.get("VolumeTracker", &pluginAlreadyRegisteredFlag);
        if (!pluginAlreadyRegisteredFlag) {
            plugin->init(simulator);
        }
        return static_cast<CompuCell3D::VolumeTrackerPlugin *>(plugin);
    }


    CompuCell3D::PixelTrackerPlugin *cc3d_ensure_pixel_tracker(void *simulatorPtr) {
        auto *simulator = cc3d_simulator_from_ptr(simulatorPtr);
        if (!simulator) {
            return nullptr;
        }

        bool pluginAlreadyRegisteredFlag = false;
        auto *plugin = CompuCell3D::Simulator::pluginManager.get("PixelTracker", &pluginAlreadyRegisteredFlag);
        if (!pluginAlreadyRegisteredFlag) {
            plugin->init(simulator);
        }
        return static_cast<CompuCell3D::PixelTrackerPlugin *>(plugin);
    }

    CompuCell3D::BoundaryPixelTrackerPlugin *cc3d_ensure_boundary_pixel_tracker(void *simulatorPtr) {
        auto *simulator = cc3d_simulator_from_ptr(simulatorPtr);
        if (!simulator) {
            return nullptr;
        }

        bool pluginAlreadyRegisteredFlag = false;
        auto *plugin = CompuCell3D::Simulator::pluginManager.get("BoundaryPixelTracker", &pluginAlreadyRegisteredFlag);
        if (!pluginAlreadyRegisteredFlag) {
            plugin->init(simulator);
        }
        return static_cast<CompuCell3D::BoundaryPixelTrackerPlugin *>(plugin);
    }

    CompuCell3D::NeighborTrackerPlugin *cc3d_ensure_neighbor_tracker(void *simulatorPtr) {
        auto *simulator = cc3d_simulator_from_ptr(simulatorPtr);
        if (!simulator) {
            return nullptr;
        }

        bool pluginAlreadyRegisteredFlag = false;
        auto *plugin = CompuCell3D::Simulator::pluginManager.get("NeighborTracker", &pluginAlreadyRegisteredFlag);
        if (!pluginAlreadyRegisteredFlag) {
            plugin->init(simulator);
        }
        return static_cast<CompuCell3D::NeighborTrackerPlugin *>(plugin);
    }

    AdhesionFlexBridge *cc3d_ensure_adhesion_flex(void *simulatorPtr) {
        auto *simulator = cc3d_simulator_from_ptr(simulatorPtr);
        if (!simulator) {
            return nullptr;
        }

        bool pluginAlreadyRegisteredFlag = false;
        auto *plugin = CompuCell3D::Simulator::pluginManager.get("AdhesionFlex", &pluginAlreadyRegisteredFlag);
        if (!pluginAlreadyRegisteredFlag) {
            auto *adhesionFlexXML = simulator->getCC3DModuleData("Plugin", "AdhesionFlex");
            plugin->init(simulator, adhesionFlexXML);
        }
        static AdhesionFlexBridge bridge{};
        bridge.callbacks = cc3d_get_adhesion_flex_callbacks(simulatorPtr);
        return bridge.callbacks ? &bridge : nullptr;
    }

    const CC3DLengthConstraintCallbacksV1 *cc3d_ensure_length_constraint(void *simulatorPtr) {
        auto *simulator = cc3d_simulator_from_ptr(simulatorPtr);
        if (!simulator) return nullptr;
        bool alreadyRegistered = false;
        auto *plugin = CompuCell3D::Simulator::pluginManager.get("LengthConstraint", &alreadyRegistered);
        if (!alreadyRegistered) {
            auto *xml = simulator->getCC3DModuleData("Plugin", "LengthConstraint");
            plugin->init(simulator, xml);
        }
        return cc3d_get_length_constraint_callbacks(simulatorPtr);
    }

    void cc3d_length_constraint_set(void *simulatorPtr, CC3DCellHandle cellHandle,
                                    double lambdaLength, double targetLength,
                                    double minorTargetLength) {
        auto *callbacks = cc3d_ensure_length_constraint(simulatorPtr);
        if (callbacks && callbacks->set_fn) callbacks->set_fn(callbacks->plugin_instance, cellHandle, lambdaLength, targetLength, minorTargetLength);
    }

    double cc3d_length_constraint_lambda(void *simulatorPtr, CC3DCellHandle cellHandle) {
        auto *callbacks = cc3d_ensure_length_constraint(simulatorPtr);
        return callbacks && callbacks->get_lambda_fn ? callbacks->get_lambda_fn(callbacks->plugin_instance, cellHandle) : 0.0;
    }

    double cc3d_length_constraint_target(void *simulatorPtr, CC3DCellHandle cellHandle) {
        auto *callbacks = cc3d_ensure_length_constraint(simulatorPtr);
        return callbacks && callbacks->get_target_fn ? callbacks->get_target_fn(callbacks->plugin_instance, cellHandle) : 0.0;
    }

    double cc3d_length_constraint_minor_target(void *simulatorPtr, CC3DCellHandle cellHandle) {
        auto *callbacks = cc3d_ensure_length_constraint(simulatorPtr);
        return callbacks && callbacks->get_minor_target_fn ? callbacks->get_minor_target_fn(callbacks->plugin_instance, cellHandle) : 0.0;
    }

    FocalPointPlasticityBridge *cc3d_ensure_focal_point_plasticity(void *simulatorPtr) {
        auto *simulator = cc3d_simulator_from_ptr(simulatorPtr);
        if (!simulator) return nullptr;
        bool alreadyRegistered = false;
        auto *plugin = CompuCell3D::Simulator::pluginManager.get("FocalPointPlasticity", &alreadyRegistered);
        if (!alreadyRegistered) {
            auto *xml = simulator->getCC3DModuleData("Plugin", "FocalPointPlasticity");
            plugin->init(simulator, xml);
        }
        static FocalPointPlasticityBridge bridge{};
        bridge.callbacks = cc3d_get_focal_point_plasticity_callbacks(simulatorPtr);
        return bridge.callbacks ? &bridge : nullptr;
}

    const CC3DClusterSurfaceCallbacksV1 *cc3d_ensure_cluster_surface(void *simulatorPtr) {
        auto *simulator = cc3d_simulator_from_ptr(simulatorPtr);
        if (!simulator) return nullptr;
        bool alreadyRegistered = false;
        auto *plugin = CompuCell3D::Simulator::pluginManager.get("ClusterSurface", &alreadyRegistered);
        if (!alreadyRegistered) {
            plugin->init(simulator, simulator->getCC3DModuleData("Plugin", "ClusterSurface"));
        }
        return cc3d_get_cluster_surface_callbacks(simulatorPtr);
    }

    void cc3d_cluster_surface_set(void *simulatorPtr, CC3DCellHandle cellHandle,
                                  float targetClusterSurface, float lambdaClusterSurface) {
        auto *callbacks = cc3d_ensure_cluster_surface(simulatorPtr);
        if (callbacks && callbacks->set_fn && cellHandle) {
            callbacks->set_fn(callbacks->plugin_instance, cellHandle, targetClusterSurface, lambdaClusterSurface);
        }
    }

    void cc3d_cluster_surface_get(void *simulatorPtr, CC3DCellHandle cellHandle,
                                  float *targetClusterSurface, float *lambdaClusterSurface) {
        if (!targetClusterSurface || !lambdaClusterSurface || !cellHandle) return;
        auto *callbacks = cc3d_ensure_cluster_surface(simulatorPtr);
        if (callbacks && callbacks->get_fn) {
            callbacks->get_fn(callbacks->plugin_instance, cellHandle, targetClusterSurface, lambdaClusterSurface);
        }
    }

    const CC3DClusterSurfaceTrackerCallbacksV1 *cc3d_ensure_cluster_surface_tracker(void *simulatorPtr) {
        auto *simulator = cc3d_simulator_from_ptr(simulatorPtr);
        if (!simulator) return nullptr;
        bool alreadyRegistered = false;
        auto *plugin = CompuCell3D::Simulator::pluginManager.get("ClusterSurfaceTracker", &alreadyRegistered);
        if (!alreadyRegistered) {
            plugin->init(simulator, simulator->getCC3DModuleData("Plugin", "ClusterSurfaceTracker"));
        }
        return cc3d_get_cluster_surface_tracker_callbacks(simulatorPtr);
    }

    void cc3d_cluster_surface_tracker_update(void *simulatorPtr, long clusterId) {
        auto *callbacks = cc3d_ensure_cluster_surface_tracker(simulatorPtr);
        if (callbacks && callbacks->update_cluster_surface_fn) {
            callbacks->update_cluster_surface_fn(callbacks->plugin_instance, clusterId);
        }
    }

    uint8_t cc3d_reassign_cluster_id(void *simulatorPtr, CC3DCellHandle cellHandle, long clusterId) {
        if (!simulatorPtr || !cellHandle) return 0u;
        auto *simulator = cc3d_simulator_from_ptr(simulatorPtr);
        if (!simulator) return 0u;
        auto *cell = static_cast<CompuCell3D::CellG *>(cellHandle);
        const long oldClusterId = cell->clusterId;
        const bool reassigned = simulator->getPotts()->getCellInventory().reassignClusterId(cell, clusterId);
        cc3d_cluster_surface_tracker_update(simulatorPtr, oldClusterId);
        cc3d_cluster_surface_tracker_update(simulatorPtr, clusterId);
        return reassigned ? 1u : 0u;
    }

    const CC3DMitosisCallbacksV1 *cc3d_ensure_mitosis(void *simulatorPtr) {
        auto *simulator = cc3d_simulator_from_ptr(simulatorPtr);
        if (!simulator) return nullptr;
        bool alreadyRegistered = false;
        auto *steppable = CompuCell3D::Simulator::steppableManager.get("Mitosis", &alreadyRegistered);
        if (!alreadyRegistered) steppable->init(simulator);
        return cc3d_get_mitosis_callbacks(simulatorPtr);
    }

    void cc3d_mitosis_set_position_flag(void *simulatorPtr, int flag) {
        auto *callbacks = cc3d_ensure_mitosis(simulatorPtr);
        if (callbacks && callbacks->set_parent_child_position_fn) callbacks->set_parent_child_position_fn(callbacks->steppable_instance, flag);
    }
    int cc3d_mitosis_get_position_flag(void *simulatorPtr) {
        auto *callbacks = cc3d_ensure_mitosis(simulatorPtr);
        return callbacks && callbacks->get_parent_child_position_fn ? callbacks->get_parent_child_position_fn(callbacks->steppable_instance) : 0;
    }
#define CC3D_MITOSIS_CELL_CALL(name, member) \
    uint8_t name(void *simulatorPtr, CC3DCellHandle cell) { auto *callbacks = cc3d_ensure_mitosis(simulatorPtr); return callbacks && callbacks->member && cell ? callbacks->member(callbacks->steppable_instance, cell) : 0u; }
    CC3D_MITOSIS_CELL_CALL(cc3d_mitosis_divide_random, divide_random_fn)
    CC3D_MITOSIS_CELL_CALL(cc3d_mitosis_divide_major, divide_major_axis_fn)
    CC3D_MITOSIS_CELL_CALL(cc3d_mitosis_divide_minor, divide_minor_axis_fn)
#undef CC3D_MITOSIS_CELL_CALL
    uint8_t cc3d_mitosis_divide_orientation(void *simulatorPtr, CC3DCellHandle cell, double nx, double ny, double nz) {
        auto *callbacks = cc3d_ensure_mitosis(simulatorPtr);
        return callbacks && callbacks->divide_orientation_fn && cell ? callbacks->divide_orientation_fn(callbacks->steppable_instance, cell, nx, ny, nz) : 0u;
    }
#define CC3D_MITOSIS_CLUSTER_CALL(name, member) \
    uint8_t name(void *simulatorPtr, long clusterId) { auto *callbacks = cc3d_ensure_mitosis(simulatorPtr); return callbacks && callbacks->member ? callbacks->member(callbacks->steppable_instance, clusterId) : 0u; }
    CC3D_MITOSIS_CLUSTER_CALL(cc3d_mitosis_divide_cluster_random, divide_cluster_random_fn)
    CC3D_MITOSIS_CLUSTER_CALL(cc3d_mitosis_divide_cluster_major, divide_cluster_major_axis_fn)
    CC3D_MITOSIS_CLUSTER_CALL(cc3d_mitosis_divide_cluster_minor, divide_cluster_minor_axis_fn)
#undef CC3D_MITOSIS_CLUSTER_CALL
    uint8_t cc3d_mitosis_divide_cluster_orientation(void *simulatorPtr, long clusterId, double nx, double ny, double nz) {
        auto *callbacks = cc3d_ensure_mitosis(simulatorPtr);
        return callbacks && callbacks->divide_cluster_orientation_fn ? callbacks->divide_cluster_orientation_fn(callbacks->steppable_instance, clusterId, nx, ny, nz) : 0u;
    }
    CC3DCellViewV1 cc3d_mitosis_parent(void *simulatorPtr) {
        auto *callbacks = cc3d_ensure_mitosis(simulatorPtr);
        return callbacks && callbacks->parent_cell_fn ? cc3d_cell_view_from_cell(static_cast<CompuCell3D::CellG *>(callbacks->parent_cell_fn(callbacks->steppable_instance))) : CC3DCellViewV1{};
    }
    CC3DCellViewV1 cc3d_mitosis_child(void *simulatorPtr) {
        auto *callbacks = cc3d_ensure_mitosis(simulatorPtr);
        return callbacks && callbacks->child_cell_fn ? cc3d_cell_view_from_cell(static_cast<CompuCell3D::CellG *>(callbacks->child_cell_fn(callbacks->steppable_instance))) : CC3DCellViewV1{};
    }

    void cc3d_mitosis_clone_attributes(void *simulatorPtr, CC3DCellHandle sourceHandle, CC3DCellHandle targetHandle) {
        auto *source = static_cast<CompuCell3D::CellG *>(sourceHandle);
        auto *target = static_cast<CompuCell3D::CellG *>(targetHandle);
        if (!source || !target) return;

        // Keep this list synchronized with SteppableBasePy.clonable_attribute_names.
        target->lambdaVolume = source->lambdaVolume;
        target->targetVolume = source->targetVolume;
        target->targetSurface = source->targetSurface;
        target->lambdaSurface = source->lambdaSurface;
        target->targetClusterSurface = source->targetClusterSurface;
        target->lambdaClusterSurface = source->lambdaClusterSurface;
        target->type = source->type;
        target->lambdaVecX = source->lambdaVecX;
        target->lambdaVecY = source->lambdaVecY;
        target->lambdaVecZ = source->lambdaVecZ;
        target->fluctAmpl = source->fluctAmpl;

        // Copy plugin state currently covered by the C ABI.
        auto *simulator = cc3d_simulator_from_ptr(simulatorPtr);
        if (simulator && CompuCell3D::Simulator::pluginManager.isLoaded("AdhesionFlex")) {
            auto *adhesion = cc3d_ensure_adhesion_flex(simulatorPtr);
            if (adhesion) {
            std::vector<float> values;
            const size_t count = adhesion->getAdhesionMoleculeNameVecSize();
            values.reserve(count);
            for (size_t i = 0; i < count; ++i) values.push_back(adhesion->getAdhesionMoleculeDensityByIndex(source, static_cast<int>(i)));
            adhesion->assignNewAdhesionMoleculeDensityVector(target, values);
            }
        }
        if (simulator && CompuCell3D::Simulator::pluginManager.isLoaded("PolarizationVector")) {
            auto *polarization = cc3d_ensure_polarization_vector(simulatorPtr);
            if (polarization) {
            float x = 0.0f, y = 0.0f, z = 0.0f;
            if (polarization->get_fn && polarization->get_fn(polarization->plugin_instance, source, &x, &y, &z) && polarization->set_fn) {
                polarization->set_fn(polarization->plugin_instance, target, x, y, z);
            }
            }
        }
        if (simulator && CompuCell3D::Simulator::pluginManager.isLoaded("CellOrientation")) {
            auto *orientation = cc3d_ensure_cell_orientation(simulatorPtr);
            if (orientation) {
            if (orientation->get_lambda_fn && orientation->set_lambda_fn) {
                orientation->set_lambda_fn(orientation->plugin_instance, target,
                                           orientation->get_lambda_fn(orientation->plugin_instance, source));
            }
            }
        }
        if (simulator && CompuCell3D::Simulator::pluginManager.isLoaded("LengthConstraint")) {
            auto *length = cc3d_ensure_length_constraint(simulatorPtr);
            if (length) {
            if (length->get_lambda_fn && length->get_target_fn && length->get_minor_target_fn && length->set_fn) {
                length->set_fn(length->plugin_instance, target,
                               length->get_lambda_fn(length->plugin_instance, source),
                               length->get_target_fn(length->plugin_instance, source),
                               length->get_minor_target_fn(length->plugin_instance, source));
            }
            }
        }
        if (simulator && CompuCell3D::Simulator::pluginManager.isLoaded("Chemotaxis")) {
            auto *chemotaxis = cc3d_ensure_chemotaxis(simulatorPtr);
            if (chemotaxis) {
            if (chemotaxis->field_count_fn && chemotaxis->field_name_fn && chemotaxis->get_data_fn && chemotaxis->add_data_fn) {
                for (size_t i = 0; i < chemotaxis->field_count_fn(chemotaxis->plugin_instance, source); ++i) {
                    const char *field = chemotaxis->field_name_fn(chemotaxis->plugin_instance, source, i);
                    void *sourceData = chemotaxis->get_data_fn(chemotaxis->plugin_instance, source, field);
                    void *targetData = chemotaxis->add_data_fn(chemotaxis->plugin_instance, target, field);
                    if (!sourceData || !targetData) continue;
                    if (chemotaxis->data_get_lambda_fn && chemotaxis->data_set_lambda_fn) chemotaxis->data_set_lambda_fn(targetData, chemotaxis->data_get_lambda_fn(sourceData));
                    if (chemotaxis->data_get_saturation_fn && chemotaxis->data_set_saturation_fn) chemotaxis->data_set_saturation_fn(targetData, chemotaxis->data_get_saturation_fn(sourceData));
                    if (chemotaxis->data_get_formula_name_fn && chemotaxis->data_set_formula_name_fn) chemotaxis->data_set_formula_name_fn(targetData, chemotaxis->data_get_formula_name_fn(sourceData));
                    if (chemotaxis->data_get_toward_count_fn && chemotaxis->data_get_toward_type_fn && chemotaxis->data_assign_toward_types_fn) {
                        std::vector<int> toward;
                        for (size_t j = 0; j < chemotaxis->data_get_toward_count_fn(sourceData); ++j) toward.push_back(chemotaxis->data_get_toward_type_fn(sourceData, j));
                        chemotaxis->data_assign_toward_types_fn(targetData, toward.data(), toward.size());
                    }
                }
            }
            }
        }

        // The following Python clone sections are deliberately excluded from this ABI:
        // ContactOrientation, ContactLocalProduct, Polarization23,
        // ConnectivityGlobal/LocalFlex, SBML/dict state, MaBoSS state, and
        // FocalPointPlasticity links remain intentionally unsupported here.
    }

    const CC3DOrientedGrowthCallbacksV1 *cc3d_ensure_oriented_growth(void *simulatorPtr) {
        auto *simulator = cc3d_simulator_from_ptr(simulatorPtr);
        if (!simulator) return nullptr;
        bool alreadyRegistered = false;
        auto *plugin = CompuCell3D::Simulator::pluginManager.get("OrientedGrowth", &alreadyRegistered);
        if (!alreadyRegistered) plugin->init(simulator, simulator->getCC3DModuleData("Plugin", "OrientedGrowth"));
        return cc3d_get_oriented_growth_callbacks(simulatorPtr);
    }

    void cc3d_oriented_growth_set_constraint_width(void *simulatorPtr, CC3DCellHandle cell, float value) {
        auto *callbacks = cc3d_ensure_oriented_growth(simulatorPtr);
        if (callbacks && callbacks->set_constraint_width_fn && cell) callbacks->set_constraint_width_fn(callbacks->plugin_instance, cell, value);
    }
    void cc3d_oriented_growth_set_axis(void *simulatorPtr, CC3DCellHandle cell, float x, float y) {
        auto *callbacks = cc3d_ensure_oriented_growth(simulatorPtr);
        if (callbacks && callbacks->set_elongation_axis_fn && cell) callbacks->set_elongation_axis_fn(callbacks->plugin_instance, cell, x, y);
    }
    void cc3d_oriented_growth_set_enabled(void *simulatorPtr, CC3DCellHandle cell, uint8_t enabled) {
        auto *callbacks = cc3d_ensure_oriented_growth(simulatorPtr);
        if (callbacks && callbacks->set_elongation_enabled_fn && cell) callbacks->set_elongation_enabled_fn(callbacks->plugin_instance, cell, enabled);
    }
    float cc3d_oriented_growth_get_axis_x(void *simulatorPtr, CC3DCellHandle cell) {
        auto *callbacks = cc3d_ensure_oriented_growth(simulatorPtr);
        return callbacks && callbacks->get_elongation_axis_x_fn && cell ? callbacks->get_elongation_axis_x_fn(callbacks->plugin_instance, cell) : 0.0f;
    }
    float cc3d_oriented_growth_get_axis_y(void *simulatorPtr, CC3DCellHandle cell) {
        auto *callbacks = cc3d_ensure_oriented_growth(simulatorPtr);
        return callbacks && callbacks->get_elongation_axis_y_fn && cell ? callbacks->get_elongation_axis_y_fn(callbacks->plugin_instance, cell) : 0.0f;
    }
    uint8_t cc3d_oriented_growth_get_enabled(void *simulatorPtr, CC3DCellHandle cell) {
        auto *callbacks = cc3d_ensure_oriented_growth(simulatorPtr);
        return callbacks && callbacks->get_elongation_enabled_fn && cell ? callbacks->get_elongation_enabled_fn(callbacks->plugin_instance, cell) : 0u;
    }

    const CC3DMomentOfInertiaCallbacksV1 *cc3d_ensure_moment_of_inertia(void *simulatorPtr) {
        auto *simulator = cc3d_simulator_from_ptr(simulatorPtr);
        if (!simulator) return nullptr;
        bool alreadyRegistered = false;
        auto *plugin = CompuCell3D::Simulator::pluginManager.get("MomentOfInertia", &alreadyRegistered);
        if (!alreadyRegistered) plugin->init(simulator, simulator->getCC3DModuleData("Plugin", "MomentOfInertia"));
        return cc3d_get_moment_of_inertia_callbacks(simulatorPtr);
    }

    size_t cc3d_moment_of_inertia_get_semiaxes(void *s, CC3DCellHandle c, double *v, size_t n) {
        auto *callbacks = cc3d_ensure_moment_of_inertia(s);
        return callbacks && callbacks->get_semiaxes_fn && c && v ? callbacks->get_semiaxes_fn(callbacks->plugin_instance, c, v, n) : 0u;
    }
    size_t cc3d_moment_of_inertia_get_semiaxes_xy(void *s, CC3DCellHandle c, double *v, size_t n) {
        auto *callbacks = cc3d_ensure_moment_of_inertia(s);
        return callbacks && callbacks->get_semiaxes_xy_fn && c && v ? callbacks->get_semiaxes_xy_fn(callbacks->plugin_instance, c, v, n) : 0u;
    }
    size_t cc3d_moment_of_inertia_get_semiaxes_xz(void *s, CC3DCellHandle c, double *v, size_t n) {
        auto *callbacks = cc3d_ensure_moment_of_inertia(s);
        return callbacks && callbacks->get_semiaxes_xz_fn && c && v ? callbacks->get_semiaxes_xz_fn(callbacks->plugin_instance, c, v, n) : 0u;
    }
    size_t cc3d_moment_of_inertia_get_semiaxes_yz(void *s, CC3DCellHandle c, double *v, size_t n) {
        auto *callbacks = cc3d_ensure_moment_of_inertia(s);
        return callbacks && callbacks->get_semiaxes_yz_fn && c && v ? callbacks->get_semiaxes_yz_fn(callbacks->plugin_instance, c, v, n) : 0u;
    }
    size_t cc3d_moment_of_inertia_get_semiaxes_3d(void *s, CC3DCellHandle c, double *v, size_t n) {
        auto *callbacks = cc3d_ensure_moment_of_inertia(s);
        return callbacks && callbacks->get_semiaxes_3d_fn && c && v ? callbacks->get_semiaxes_3d_fn(callbacks->plugin_instance, c, v, n) : 0u;
    }

    const CC3DPolarizationVectorCallbacksV1 *cc3d_ensure_polarization_vector(void *simulatorPtr) {
        auto *simulator = cc3d_simulator_from_ptr(simulatorPtr);
        if (!simulator) return nullptr;
        bool alreadyRegistered = false;
        auto *plugin = CompuCell3D::Simulator::pluginManager.get("PolarizationVector", &alreadyRegistered);
        if (!alreadyRegistered) plugin->init(simulator, simulator->getCC3DModuleData("Plugin", "PolarizationVector"));
        return cc3d_get_polarization_vector_callbacks(simulatorPtr);
    }

    const CC3DCellOrientationCallbacksV1 *cc3d_ensure_cell_orientation(void *simulatorPtr) {
        auto *simulator = cc3d_simulator_from_ptr(simulatorPtr);
        if (!simulator) return nullptr;
        bool alreadyRegistered = false;
        auto *plugin = CompuCell3D::Simulator::pluginManager.get("CellOrientation", &alreadyRegistered);
        if (!alreadyRegistered) plugin->init(simulator, simulator->getCC3DModuleData("Plugin", "CellOrientation"));
        return cc3d_get_cell_orientation_callbacks(simulatorPtr);
    }

    void cc3d_cell_orientation_set_lambda(void *simulatorPtr, CC3DCellHandle cell, double value) {
        auto *api = cc3d_ensure_cell_orientation(simulatorPtr);
        if (api && api->set_lambda_fn) api->set_lambda_fn(api->plugin_instance, cell, value);
    }
    double cc3d_cell_orientation_get_lambda(void *simulatorPtr, CC3DCellHandle cell) {
        auto *api = cc3d_ensure_cell_orientation(simulatorPtr);
        return api && api->get_lambda_fn ? api->get_lambda_fn(api->plugin_instance, cell) : 0.0;
    }

    void cc3d_polarization_set(void *simulatorPtr, CC3DCellHandle cell, float x, float y, float z) {
        auto *api = cc3d_ensure_polarization_vector(simulatorPtr);
        if (api && api->set_fn) api->set_fn(api->plugin_instance, cell, x, y, z);
    }
    uint8_t cc3d_polarization_get(void *simulatorPtr, CC3DCellHandle cell, float *x, float *y, float *z) {
        auto *api = cc3d_ensure_polarization_vector(simulatorPtr);
        return api && api->get_fn ? api->get_fn(api->plugin_instance, cell, x, y, z) : 0u;
    }

    const CC3DChemotaxisCallbacksV1 *cc3d_ensure_chemotaxis(void *simulatorPtr) {
        auto *simulator = cc3d_simulator_from_ptr(simulatorPtr);
        if (!simulator) return nullptr;
        bool alreadyRegistered = false;
        auto *plugin = CompuCell3D::Simulator::pluginManager.get("Chemotaxis", &alreadyRegistered);
        if (!alreadyRegistered) plugin->init(simulator, simulator->getCC3DModuleData("Plugin", "Chemotaxis"));
        return cc3d_get_chemotaxis_callbacks(simulatorPtr);
    }
    size_t cc3d_chemotaxis_field_count(void *simulatorPtr, CC3DCellHandle cell) {
        auto *api = cc3d_ensure_chemotaxis(simulatorPtr);
        return api && api->field_count_fn ? api->field_count_fn(api->plugin_instance, cell) : 0u;
    }
    const char *cc3d_chemotaxis_field_name(void *simulatorPtr, CC3DCellHandle cell, size_t index) {
        auto *api = cc3d_ensure_chemotaxis(simulatorPtr);
        return api && api->field_name_fn ? api->field_name_fn(api->plugin_instance, cell, index) : nullptr;
    }
    CC3DChemotaxisDataAPIV1 cc3d_chemotaxis_data_api(void *simulatorPtr, CC3DCellHandle cell, const char *name, bool add) {
        CC3DChemotaxisDataAPIV1 result{};
        auto *api = cc3d_ensure_chemotaxis(simulatorPtr);
        if (!api || !name) return result;
        result.userdata = add ? api->add_data_fn(api->plugin_instance, cell, name) : api->get_data_fn(api->plugin_instance, cell, name);
        result.get_lambda_fn = api->data_get_lambda_fn;
        result.set_lambda_fn = api->data_set_lambda_fn;
        result.get_saturation_fn = api->data_get_saturation_fn;
        result.set_saturation_fn = api->data_set_saturation_fn;
        result.get_formula_name_fn = api->data_get_formula_name_fn;
        result.set_formula_name_fn = api->data_set_formula_name_fn;
        result.get_toward_count_fn = api->data_get_toward_count_fn;
        result.get_toward_type_fn = api->data_get_toward_type_fn;
        result.assign_toward_types_fn = api->data_assign_toward_types_fn;
        return result;
    }
    CC3DChemotaxisDataAPIV1 cc3d_chemotaxis_add(void *simulatorPtr, CC3DCellHandle cell, const char *name) { return cc3d_chemotaxis_data_api(simulatorPtr, cell, name, true); }
    CC3DChemotaxisDataAPIV1 cc3d_chemotaxis_get(void *simulatorPtr, CC3DCellHandle cell, const char *name) { return cc3d_chemotaxis_data_api(simulatorPtr, cell, name, false); }

    const CC3DSecretionCallbacksV1 *cc3d_ensure_secretion(void *simulatorPtr) {
        auto *simulator = cc3d_simulator_from_ptr(simulatorPtr);
        if (!simulator) return nullptr;
        bool alreadyRegistered = false;
        auto *plugin = CompuCell3D::Simulator::pluginManager.get("Secretion", &alreadyRegistered);
        if (!alreadyRegistered) plugin->init(simulator, simulator->getCC3DModuleData("Plugin", "Secretion"));
        return cc3d_get_secretion_callbacks(simulatorPtr);
    }
#define CC3D_SECRETE_BOOL(name, member) \
    uint8_t name(void *s, CC3DCellHandle c, const char *f, float a) { auto *p = cc3d_ensure_secretion(s); return p && p->member ? p->member(p->plugin_instance, c, f, a) : 0u; }
#define CC3D_SECRETE_TOTAL(name, member) \
    CC3DFieldSecretorResultV1 name(void *s, CC3DCellHandle c, const char *f, float a) { auto *p = cc3d_ensure_secretion(s); return p && p->member ? p->member(p->plugin_instance, c, f, a) : CC3DFieldSecretorResultV1{0, 0.0f}; }
    CC3D_SECRETE_BOOL(cc3d_secrete_inside, secrete_inside_fn)
    CC3D_SECRETE_TOTAL(cc3d_secrete_inside_total, secrete_inside_total_fn)
    CC3D_SECRETE_BOOL(cc3d_secrete_inside_constant, secrete_inside_constant_fn)
    CC3D_SECRETE_TOTAL(cc3d_secrete_inside_constant_total, secrete_inside_constant_total_fn)
    CC3D_SECRETE_BOOL(cc3d_secrete_inside_boundary, secrete_inside_boundary_fn)
    CC3D_SECRETE_TOTAL(cc3d_secrete_inside_boundary_total, secrete_inside_boundary_total_fn)
    uint8_t cc3d_secrete_inside_boundary_contact(void *s, CC3DCellHandle c, const char *f, float a, const unsigned char *t, size_t n) { auto *p = cc3d_ensure_secretion(s); return p && p->secrete_inside_boundary_contact_fn ? p->secrete_inside_boundary_contact_fn(p->plugin_instance, c, f, a, t, n) : 0u; }
    CC3DFieldSecretorResultV1 cc3d_secrete_inside_boundary_contact_total(void *s, CC3DCellHandle c, const char *f, float a, const unsigned char *t, size_t n) { auto *p = cc3d_ensure_secretion(s); return p && p->secrete_inside_boundary_contact_total_fn ? p->secrete_inside_boundary_contact_total_fn(p->plugin_instance, c, f, a, t, n) : CC3DFieldSecretorResultV1{0, 0.0f}; }
    CC3D_SECRETE_BOOL(cc3d_secrete_outside_boundary, secrete_outside_boundary_fn)
    CC3D_SECRETE_TOTAL(cc3d_secrete_outside_boundary_total, secrete_outside_boundary_total_fn)
    uint8_t cc3d_secrete_outside_boundary_contact(void *s, CC3DCellHandle c, const char *f, float a, const unsigned char *t, size_t n) { auto *p = cc3d_ensure_secretion(s); return p && p->secrete_outside_boundary_contact_fn ? p->secrete_outside_boundary_contact_fn(p->plugin_instance, c, f, a, t, n) : 0u; }
    CC3DFieldSecretorResultV1 cc3d_secrete_outside_boundary_contact_total(void *s, CC3DCellHandle c, const char *f, float a, const unsigned char *t, size_t n) { auto *p = cc3d_ensure_secretion(s); return p && p->secrete_outside_boundary_contact_total_fn ? p->secrete_outside_boundary_contact_total_fn(p->plugin_instance, c, f, a, t, n) : CC3DFieldSecretorResultV1{0, 0.0f}; }
    CC3D_SECRETE_BOOL(cc3d_secrete_com, secrete_com_fn)
    CC3D_SECRETE_TOTAL(cc3d_secrete_com_total, secrete_com_total_fn)
#define CC3D_UPTAKE_BOOL(name, member) \
    uint8_t name(void *s, CC3DCellHandle c, const char *f, float max, float rel) { auto *p = cc3d_ensure_secretion(s); return p && p->member ? p->member(p->plugin_instance, c, f, max, rel) : 0u; }
#define CC3D_UPTAKE_TOTAL(name, member) \
    CC3DFieldSecretorResultV1 name(void *s, CC3DCellHandle c, const char *f, float max, float rel) { auto *p = cc3d_ensure_secretion(s); return p && p->member ? p->member(p->plugin_instance, c, f, max, rel) : CC3DFieldSecretorResultV1{0, 0.0f}; }
    CC3D_UPTAKE_BOOL(cc3d_uptake_inside, uptake_inside_fn)
    CC3D_UPTAKE_TOTAL(cc3d_uptake_inside_total, uptake_inside_total_fn)
    uint8_t cc3d_uptake_boundary(void *s, CC3DCellHandle c, const char *f, float max, float rel) { auto *p = cc3d_ensure_secretion(s); return p && p->uptake_boundary_fn ? p->uptake_boundary_fn(p->plugin_instance, c, f, max, rel) : 0u; }
    CC3DFieldSecretorResultV1 cc3d_uptake_boundary_total(void *s, CC3DCellHandle c, const char *f, float max, float rel) { auto *p = cc3d_ensure_secretion(s); return p && p->uptake_boundary_total_fn ? p->uptake_boundary_total_fn(p->plugin_instance, c, f, max, rel) : CC3DFieldSecretorResultV1{0, 0.0f}; }
    uint8_t cc3d_uptake_outside_boundary(void *s, CC3DCellHandle c, const char *f, float max, float rel) { auto *p = cc3d_ensure_secretion(s); return p && p->uptake_outside_boundary_fn ? p->uptake_outside_boundary_fn(p->plugin_instance, c, f, max, rel) : 0u; }
    CC3DFieldSecretorResultV1 cc3d_uptake_outside_boundary_total(void *s, CC3DCellHandle c, const char *f, float max, float rel) { auto *p = cc3d_ensure_secretion(s); return p && p->uptake_outside_boundary_total_fn ? p->uptake_outside_boundary_total_fn(p->plugin_instance, c, f, max, rel) : CC3DFieldSecretorResultV1{0, 0.0f}; }
    float cc3d_secretion_amount_seen(void *s, CC3DCellHandle c, const char *f) { auto *p = cc3d_ensure_secretion(s); return p && p->amount_seen_fn ? p->amount_seen_fn(p->plugin_instance, c, f) : 0.0f; }
    float cc3d_secretion_field_integral(void *s, const char *f) { auto *p = cc3d_ensure_secretion(s); return p && p->field_integral_fn ? p->field_integral_fn(p->plugin_instance, f) : 0.0f; }
    void cc3d_secretion_single(void *s, size_t i) { auto *p = cc3d_ensure_secretion(s); if (p && p->secrete_single_field_fn) p->secrete_single_field_fn(p->plugin_instance, i); }
    void cc3d_secretion_contact_single(void *s, size_t i) { auto *p = cc3d_ensure_secretion(s); if (p && p->secrete_contact_single_field_fn) p->secrete_contact_single_field_fn(p->plugin_instance, i); }
    void cc3d_secretion_constant_single(void *s, size_t i) { auto *p = cc3d_ensure_secretion(s); if (p && p->secrete_constant_single_field_fn) p->secrete_constant_single_field_fn(p->plugin_instance, i); }
#undef CC3D_SECRETE_BOOL
#undef CC3D_SECRETE_TOTAL
#undef CC3D_UPTAKE_BOOL
#undef CC3D_UPTAKE_TOTAL

    void cc3d_fpp_create(void *simulatorPtr, CC3DCellHandle a, CC3DCellHandle b, double lambda, double target, double max) {
        auto *p = cc3d_ensure_focal_point_plasticity(simulatorPtr);
        if (p) p->createFocalPointPlasticityLink(static_cast<CompuCell3D::CellG *>(a), static_cast<CompuCell3D::CellG *>(b), lambda, target, max);
    }
    void cc3d_fpp_create_internal(void *simulatorPtr, CC3DCellHandle a, CC3DCellHandle b, double lambda, double target, double max) {
        auto *p = cc3d_ensure_focal_point_plasticity(simulatorPtr);
        if (p) p->createInternalFocalPointPlasticityLink(static_cast<CompuCell3D::CellG *>(a), static_cast<CompuCell3D::CellG *>(b), lambda, target, max);
    }
    void cc3d_fpp_delete(void *simulatorPtr, CC3DCellHandle a, CC3DCellHandle b) {
        auto *p = cc3d_ensure_focal_point_plasticity(simulatorPtr);
        if (p) p->deleteFocalPointPlasticityLink(static_cast<CompuCell3D::CellG *>(a), static_cast<CompuCell3D::CellG *>(b));
    }
    void cc3d_fpp_delete_internal(void *simulatorPtr, CC3DCellHandle a, CC3DCellHandle b) {
        auto *p = cc3d_ensure_focal_point_plasticity(simulatorPtr);
        if (p) p->deleteInternalFocalPointPlasticityLink(static_cast<CompuCell3D::CellG *>(a), static_cast<CompuCell3D::CellG *>(b));
    }
    void cc3d_fpp_set(void *simulatorPtr, CC3DCellHandle a, CC3DCellHandle b, double lambda, double target, double max) {
        auto *p = cc3d_ensure_focal_point_plasticity(simulatorPtr);
        if (p) p->setFocalPointPlasticityParameters(static_cast<CompuCell3D::CellG *>(a), static_cast<CompuCell3D::CellG *>(b), lambda, target, max);
    }
    double cc3d_fpp_lambda(void *simulatorPtr, CC3DCellHandle a, CC3DCellHandle b) {
        auto *p = cc3d_ensure_focal_point_plasticity(simulatorPtr);
        return p ? p->getPlasticityParametersLambdaDistance(static_cast<CompuCell3D::CellG *>(a), static_cast<CompuCell3D::CellG *>(b)) : 0.0;
    }
    double cc3d_fpp_target(void *simulatorPtr, CC3DCellHandle a, CC3DCellHandle b) {
        auto *p = cc3d_ensure_focal_point_plasticity(simulatorPtr);
        return p ? p->getPlasticityParametersTargetDistance(static_cast<CompuCell3D::CellG *>(a), static_cast<CompuCell3D::CellG *>(b)) : 0.0;
    }
    int cc3d_fpp_create_anchor(void *simulatorPtr, CC3DCellHandle cell, double lambda, double target, double max, float x, float y, float z) {
        auto *p = cc3d_ensure_focal_point_plasticity(simulatorPtr);
        return p ? p->createAnchor(static_cast<CompuCell3D::CellG *>(cell), lambda, target, max, x, y, z) : -1;
    }
    void cc3d_fpp_delete_anchor(void *simulatorPtr, CC3DCellHandle cell, int id) {
        auto *p = cc3d_ensure_focal_point_plasticity(simulatorPtr);
        if (p) p->deleteAnchor(static_cast<CompuCell3D::CellG *>(cell), id);
    }
    void cc3d_fpp_set_anchor(void *simulatorPtr, CC3DCellHandle cell, int id, double lambda, double target, double max, float x, float y, float z) {
        auto *p = cc3d_ensure_focal_point_plasticity(simulatorPtr);
        if (p) p->setAnchorParameters(static_cast<CompuCell3D::CellG *>(cell), id, lambda, target, max, x, y, z);
    }

    std::vector<float> cc3d_float_vector_from_array(const float *values, size_t count) {
        if (!values || count == 0u) {
            return std::vector<float>();
        }
        return std::vector<float>(values, values + count);
    }

    size_t cc3d_adhesion_flex_molecule_count(void *simulatorPtr) {
        auto *plugin = cc3d_ensure_adhesion_flex(simulatorPtr);
        if (!plugin) {
            return 0u;
        }
        return plugin->getAdhesionMoleculeNameVec().size();
    }

    const char *cc3d_adhesion_flex_molecule_name(void *simulatorPtr, size_t index) {
        auto *plugin = cc3d_ensure_adhesion_flex(simulatorPtr);
        if (!plugin) {
            return nullptr;
        }

        static thread_local std::vector<std::string> moleculeNames;
        moleculeNames = plugin->getAdhesionMoleculeNameVec();
        if (index >= moleculeNames.size()) {
            return nullptr;
        }
        return moleculeNames[index].c_str();
    }

    float cc3d_adhesion_flex_get_density(void *simulatorPtr, CC3DCellHandle cellHandle, const char *moleculeName) {
        auto *plugin = cc3d_ensure_adhesion_flex(simulatorPtr);
        auto *cell = static_cast<CompuCell3D::CellG *>(cellHandle);
        if (!plugin || !cell || !moleculeName) {
            return 0.0f;
        }
        return plugin->getAdhesionMoleculeDensity(cell, moleculeName);
    }

    float cc3d_adhesion_flex_get_density_by_index(void *simulatorPtr, CC3DCellHandle cellHandle, size_t index) {
        auto *plugin = cc3d_ensure_adhesion_flex(simulatorPtr);
        auto *cell = static_cast<CompuCell3D::CellG *>(cellHandle);
        if (!plugin || !cell) {
            return 0.0f;
        }
        return plugin->getAdhesionMoleculeDensityByIndex(cell, static_cast<int>(index));
    }

    void cc3d_adhesion_flex_set_density(
        void *simulatorPtr, CC3DCellHandle cellHandle, const char *moleculeName, float density
    ) {
        auto *plugin = cc3d_ensure_adhesion_flex(simulatorPtr);
        auto *cell = static_cast<CompuCell3D::CellG *>(cellHandle);
        if (plugin && cell && moleculeName) {
            plugin->setAdhesionMoleculeDensity(cell, moleculeName, density);
        }
    }

    void cc3d_adhesion_flex_set_density_by_index(
        void *simulatorPtr, CC3DCellHandle cellHandle, size_t index, float density
    ) {
        auto *plugin = cc3d_ensure_adhesion_flex(simulatorPtr);
        auto *cell = static_cast<CompuCell3D::CellG *>(cellHandle);
        if (plugin && cell) {
            plugin->setAdhesionMoleculeDensityByIndex(cell, static_cast<int>(index), density);
        }
    }

    uint8_t cc3d_adhesion_flex_set_density_vector(
        void *simulatorPtr, CC3DCellHandle cellHandle, const float *densities, size_t count
    ) {
        auto *plugin = cc3d_ensure_adhesion_flex(simulatorPtr);
        auto *cell = static_cast<CompuCell3D::CellG *>(cellHandle);
        if (!plugin || !cell || !densities) {
            return 0u;
        }
        if (count != plugin->getAdhesionMoleculeNameVec().size()) {
            return 0u;
        }
        auto densityVector = cc3d_float_vector_from_array(densities, count);
        plugin->setAdhesionMoleculeDensityVector(cell, densityVector);
        return 1u;
    }

    void cc3d_adhesion_flex_assign_density_vector(
        void *simulatorPtr, CC3DCellHandle cellHandle, const float *densities, size_t count
    ) {
        auto *plugin = cc3d_ensure_adhesion_flex(simulatorPtr);
        auto *cell = static_cast<CompuCell3D::CellG *>(cellHandle);
        if (plugin && cell && densities) {
            auto densityVector = cc3d_float_vector_from_array(densities, count);
            plugin->assignNewAdhesionMoleculeDensityVector(cell, densityVector);
        }
    }

    float cc3d_adhesion_flex_get_medium_density(void *simulatorPtr, const char *moleculeName) {
        auto *plugin = cc3d_ensure_adhesion_flex(simulatorPtr);
        if (!plugin || !moleculeName) {
            return 0.0f;
        }
        return plugin->getMediumAdhesionMoleculeDensity(moleculeName);
    }

    float cc3d_adhesion_flex_get_medium_density_by_index(void *simulatorPtr, size_t index) {
        auto *plugin = cc3d_ensure_adhesion_flex(simulatorPtr);
        return plugin ? plugin->getMediumAdhesionMoleculeDensityByIndex(static_cast<int>(index)) : 0.0f;
    }

    void cc3d_adhesion_flex_set_medium_density(void *simulatorPtr, const char *moleculeName, float density) {
        auto *plugin = cc3d_ensure_adhesion_flex(simulatorPtr);
        if (plugin && moleculeName) {
            plugin->setMediumAdhesionMoleculeDensity(moleculeName, density);
        }
    }

    void cc3d_adhesion_flex_set_medium_density_by_index(void *simulatorPtr, size_t index, float density) {
        auto *plugin = cc3d_ensure_adhesion_flex(simulatorPtr);
        if (plugin) {
            plugin->setMediumAdhesionMoleculeDensityByIndex(static_cast<int>(index), density);
        }
    }

    uint8_t cc3d_adhesion_flex_set_medium_density_vector(void *simulatorPtr, const float *densities, size_t count) {
        auto *plugin = cc3d_ensure_adhesion_flex(simulatorPtr);
        if (!plugin || !densities) {
            return 0u;
        }
        if (count != plugin->getAdhesionMoleculeNameVec().size()) {
            return 0u;
        }
        auto densityVector = cc3d_float_vector_from_array(densities, count);
        plugin->setMediumAdhesionMoleculeDensityVector(densityVector);
        return 1u;
    }

    void cc3d_adhesion_flex_assign_medium_density_vector(void *simulatorPtr, const float *densities, size_t count) {
        auto *plugin = cc3d_ensure_adhesion_flex(simulatorPtr);
        if (plugin && densities) {
            auto densityVector = cc3d_float_vector_from_array(densities, count);
            plugin->assignNewMediumAdhesionMoleculeDensityVector(densityVector);
        }
    }

    CC3DCellViewV1 cc3d_cells_get(void *, CC3DCellIteratorHandle iterator) {
        auto *state = static_cast<CellIteratorState *>(iterator);

        if (!state || state->current == state->end) {
            return CC3DCellViewV1{};
        }

        return cc3d_cell_view_from_cell(state->current->second);
    }

    void cc3d_cells_next(void *, CC3DCellIteratorHandle iterator) {
        auto *state = static_cast<CellIteratorState *>(iterator);
        if (state && state->current != state->end) {
            ++state->current;
            cc3d_cells_advance_to_matching_type(state);
        }
    }

    void cc3d_cells_destroy(void *, CC3DCellIteratorHandle iterator) {
        delete static_cast<CellIteratorState *>(iterator);
    }

    CC3DCellNeighborIteratorHandle cc3d_cell_neighbors_begin(void *simulatorPtr, CC3DCellHandle cellHandle) {
        auto *plugin = cc3d_ensure_neighbor_tracker(simulatorPtr);
        auto *cell = static_cast<CompuCell3D::CellG *>(cellHandle);

        if (!plugin) {
            throw CC3DException("Compiled steppable could not initialize NeighborTracker");
        }
        if (!cell) {
            throw CC3DException("Compiled steppable neighbor iterator received a null cell");
        }

        auto &neighbors = plugin->getNeighborTrackerAccessorPtr()->get(cell->extraAttribPtr)->cellNeighbors;
        auto *state = new CellNeighborIteratorState{neighbors.begin(), neighbors.end()};
        return static_cast<CC3DCellNeighborIteratorHandle>(state);
    }

    uint8_t cc3d_cell_neighbors_valid(void *, CC3DCellNeighborIteratorHandle iterator) {
        auto *state = static_cast<CellNeighborIteratorState *>(iterator);
        return state && state->current != state->end ? 1u : 0u;
    }

    CC3DCellNeighborViewV1 cc3d_cell_neighbors_get(void *, CC3DCellNeighborIteratorHandle iterator) {
        auto *state = static_cast<CellNeighborIteratorState *>(iterator);
        if (!state || state->current == state->end) {
            return CC3DCellNeighborViewV1{};
        }

        CC3DCellNeighborViewV1 view{};
        view.neighbor = cc3d_cell_view_from_cell(state->current->neighborAddress);
        view.commonSurfaceArea = state->current->commonSurfaceArea;
        return view;
    }

    void cc3d_cell_neighbors_next(void *, CC3DCellNeighborIteratorHandle iterator) {
        auto *state = static_cast<CellNeighborIteratorState *>(iterator);
        if (state && state->current != state->end) {
            ++state->current;
        }
    }

    void cc3d_cell_neighbors_destroy(void *, CC3DCellNeighborIteratorHandle iterator) {
        delete static_cast<CellNeighborIteratorState *>(iterator);
    }


    CC3DPixelViewV1 cc3d_pixel_view_from_point(const CompuCell3D::Point3D &pt) {
        return CC3DPixelViewV1{static_cast<int>(pt.x), static_cast<int>(pt.y), static_cast<int>(pt.z)};
    }

    CC3DPixelIteratorHandle cc3d_cell_pixels_begin(void *simulatorPtr, CC3DCellHandle cellHandle) {
        auto *plugin = cc3d_ensure_pixel_tracker(simulatorPtr);
        auto *cell = static_cast<CompuCell3D::CellG *>(cellHandle);

        if (!plugin) {
            throw CC3DException("Compiled steppable could not initialize PixelTracker");
        }
        if (!cell) {
            throw CC3DException("Compiled steppable pixel iterator received a null cell");
        }

        auto &pixels = plugin->getPixelTrackerAccessorPtr()->get(cell->extraAttribPtr)->pixelSet;
        auto *state = new PixelIteratorState{pixels.begin(), pixels.end()};
        return static_cast<CC3DPixelIteratorHandle>(state);
    }

    uint8_t cc3d_cell_pixels_valid(void *, CC3DPixelIteratorHandle iterator) {
        auto *state = static_cast<PixelIteratorState *>(iterator);
        return state && state->current != state->end ? 1u : 0u;
    }

    CC3DPixelViewV1 cc3d_cell_pixels_get(void *, CC3DPixelIteratorHandle iterator) {
        auto *state = static_cast<PixelIteratorState *>(iterator);
        if (!state || state->current == state->end) {
            return CC3DPixelViewV1{0, 0, 0};
        }
        return cc3d_pixel_view_from_point(state->current->pixel);
    }

    void cc3d_cell_pixels_next(void *, CC3DPixelIteratorHandle iterator) {
        auto *state = static_cast<PixelIteratorState *>(iterator);
        if (state && state->current != state->end) {
            ++state->current;
        }
    }

    void cc3d_cell_pixels_destroy(void *, CC3DPixelIteratorHandle iterator) {
        delete static_cast<PixelIteratorState *>(iterator);
    }

    CC3DPixelIteratorHandle cc3d_boundary_pixels_begin(void *simulatorPtr, CC3DCellHandle cellHandle) {
        auto *plugin = cc3d_ensure_boundary_pixel_tracker(simulatorPtr);
        auto *cell = static_cast<CompuCell3D::CellG *>(cellHandle);

        if (!plugin) {
            throw CC3DException("Compiled steppable could not initialize BoundaryPixelTracker");
        }
        if (!cell) {
            throw CC3DException("Compiled steppable boundary pixel iterator received a null cell");
        }

        auto &pixels = plugin->getBoundaryPixelTrackerAccessorPtr()->get(cell->extraAttribPtr)->pixelSet;
        auto *state = new BoundaryPixelIteratorState{pixels.begin(), pixels.end()};
        return static_cast<CC3DPixelIteratorHandle>(state);
    }

    uint8_t cc3d_boundary_pixels_valid(void *, CC3DPixelIteratorHandle iterator) {
        auto *state = static_cast<BoundaryPixelIteratorState *>(iterator);
        return state && state->current != state->end ? 1u : 0u;
    }

    CC3DPixelViewV1 cc3d_boundary_pixels_get(void *, CC3DPixelIteratorHandle iterator) {
        auto *state = static_cast<BoundaryPixelIteratorState *>(iterator);
        if (!state || state->current == state->end) {
            return CC3DPixelViewV1{0, 0, 0};
        }
        return cc3d_pixel_view_from_point(state->current->pixel);
    }

    void cc3d_boundary_pixels_next(void *, CC3DPixelIteratorHandle iterator) {
        auto *state = static_cast<BoundaryPixelIteratorState *>(iterator);
        if (state && state->current != state->end) {
            ++state->current;
        }
    }

    void cc3d_boundary_pixels_destroy(void *, CC3DPixelIteratorHandle iterator) {
        delete static_cast<BoundaryPixelIteratorState *>(iterator);
    }

    CC3DFieldHandle cc3d_scalar_fields_find(void *simulatorPtr, const char *name) {
        auto *simulator = static_cast<CompuCell3D::Simulator *>(simulatorPtr);
        if (!simulator || !name) {
            return nullptr;
        }
        return static_cast<CC3DFieldHandle>(simulator->getConcentrationFieldByName(name));
    }

    CC3DSharedFieldHandle cc3d_shared_scalar_find(void *simulatorPtr, const char *name) {
        auto *simulator = static_cast<CompuCell3D::Simulator *>(simulatorPtr);
        if (!simulator || !name) return nullptr;
        if (auto *field = simulator->getGenericScalarFieldTypeBase(name)) return field;
        return dynamic_cast<CompuCell3D::NumpyArrayWrapper3DImpl<float> *>(
            simulator->getConcentrationFieldByName(name));
    }

    CC3DSharedFieldHandle cc3d_shared_scalar_create(
        void *simulatorPtr, const char *name, CC3DFieldPrecisionV1 precision, int padding
    ) {
        auto *simulator = static_cast<CompuCell3D::Simulator *>(simulatorPtr);
        // Padded shared fields are intentionally deferred until the Python/C++
        // storage-coordinate contract is implemented.  The first ABI version
        // exposes only fields whose logical and allocated dimensions match.
        if (!simulator || !name || padding != 0) return nullptr;
        try {
            switch (precision) {
                case CC3D_FIELD_FLOAT32: simulator->createGenericScalarField<float>(name, static_cast<size_t>(padding)); break;
                case CC3D_FIELD_FLOAT64: simulator->createGenericScalarField<double>(name, static_cast<size_t>(padding)); break;
                case CC3D_FIELD_INT16: simulator->createGenericScalarField<short>(name, static_cast<size_t>(padding)); break;
                case CC3D_FIELD_UINT16: simulator->createGenericScalarField<unsigned short>(name, static_cast<size_t>(padding)); break;
                case CC3D_FIELD_INT32: simulator->createGenericScalarField<int>(name, static_cast<size_t>(padding)); break;
                case CC3D_FIELD_UINT32: simulator->createGenericScalarField<unsigned int>(name, static_cast<size_t>(padding)); break;
                default: return nullptr;
            }
            return simulator->getGenericScalarFieldTypeBase(name);
        } catch (...) {
            return nullptr;
        }
    }

    template <typename T>
    CC3DFieldDimV1 cc3d_shared_scalar_dim_t(CompuCell3D::NumpyArrayWrapper3DImpl<T> *field, uint8_t raw) {
        if (!field) return {0, 0, 0};
        auto dim = field->getDim();
        if (!raw) {
            auto padding = field->getPaddingVec();
            dim.x -= static_cast<int>(2 * padding[0]);
            dim.y -= static_cast<int>(2 * padding[1]);
            dim.z -= static_cast<int>(2 * padding[2]);
        }
        return {dim.x, dim.y, dim.z};
    }

    CC3DFieldDimV1 cc3d_shared_scalar_dim(void *, CC3DSharedFieldHandle handle, uint8_t raw) {
        auto *base = static_cast<CompuCell3D::Field3DTypeBase *>(handle);
        if (!base) return {0, 0, 0};
#define CC3D_SHARED_SCALAR_DIM(T) \
        if (auto *field = dynamic_cast<CompuCell3D::NumpyArrayWrapper3DImpl<T> *>(base)) return cc3d_shared_scalar_dim_t(field, raw)
        CC3D_SHARED_SCALAR_DIM(float);
        CC3D_SHARED_SCALAR_DIM(double);
        CC3D_SHARED_SCALAR_DIM(short);
        CC3D_SHARED_SCALAR_DIM(unsigned short);
        CC3D_SHARED_SCALAR_DIM(int);
        CC3D_SHARED_SCALAR_DIM(unsigned int);
#undef CC3D_SHARED_SCALAR_DIM
        return {0, 0, 0};
    }

    template <typename T>
    double cc3d_shared_scalar_get_t(CompuCell3D::NumpyArrayWrapper3DImpl<T> *field, int x, int y, int z, uint8_t raw) {
        if (!field) return 0.0;
        if (raw) {
            auto padding = field->getPaddingVec();
            x -= static_cast<int>(padding[0]);
            y -= static_cast<int>(padding[1]);
            z -= static_cast<int>(padding[2]);
        }
        return static_cast<double>(field->get(CompuCell3D::Point3D(x, y, z)));
    }

    double cc3d_shared_scalar_get(void *, CC3DSharedFieldHandle handle, int x, int y, int z, uint8_t raw) {
        auto *base = static_cast<CompuCell3D::Field3DTypeBase *>(handle);
        if (!base) return 0.0;
#define CC3D_SHARED_SCALAR_GET(T) \
        if (auto *field = dynamic_cast<CompuCell3D::NumpyArrayWrapper3DImpl<T> *>(base)) return cc3d_shared_scalar_get_t(field, x, y, z, raw)
        CC3D_SHARED_SCALAR_GET(float);
        CC3D_SHARED_SCALAR_GET(double);
        CC3D_SHARED_SCALAR_GET(short);
        CC3D_SHARED_SCALAR_GET(unsigned short);
        CC3D_SHARED_SCALAR_GET(int);
        CC3D_SHARED_SCALAR_GET(unsigned int);
#undef CC3D_SHARED_SCALAR_GET
        return 0.0;
    }

    template <typename T>
    uint8_t cc3d_shared_scalar_set_t(CompuCell3D::NumpyArrayWrapper3DImpl<T> *field, int x, int y, int z, double value, uint8_t raw) {
        if (!field) return 0u;
        if (raw) {
            auto padding = field->getPaddingVec();
            x -= static_cast<int>(padding[0]);
            y -= static_cast<int>(padding[1]);
            z -= static_cast<int>(padding[2]);
        }
        field->set(CompuCell3D::Point3D(x, y, z), static_cast<T>(value));
        return 1u;
    }

    uint8_t cc3d_shared_scalar_set(void *, CC3DSharedFieldHandle handle, int x, int y, int z, double value, uint8_t raw) {
        auto *base = static_cast<CompuCell3D::Field3DTypeBase *>(handle);
        if (!base) return 0u;
#define CC3D_SHARED_SCALAR_SET(T) \
        if (auto *field = dynamic_cast<CompuCell3D::NumpyArrayWrapper3DImpl<T> *>(base)) return cc3d_shared_scalar_set_t(field, x, y, z, value, raw)
        CC3D_SHARED_SCALAR_SET(float);
        CC3D_SHARED_SCALAR_SET(double);
        CC3D_SHARED_SCALAR_SET(short);
        CC3D_SHARED_SCALAR_SET(unsigned short);
        CC3D_SHARED_SCALAR_SET(int);
        CC3D_SHARED_SCALAR_SET(unsigned int);
#undef CC3D_SHARED_SCALAR_SET
        return 0u;
    }

    CC3DSharedFieldHandle cc3d_shared_vector_find(void *simulatorPtr, const char *name) {
        auto *simulator = static_cast<CompuCell3D::Simulator *>(simulatorPtr);
        return simulator && name ? simulator->getVectorFieldByName(name) : nullptr;
    }

    CC3DSharedFieldHandle cc3d_shared_vector_create(void *simulatorPtr, const char *name, CC3DFieldPrecisionV1 precision, int padding) {
        auto *simulator = static_cast<CompuCell3D::Simulator *>(simulatorPtr);
        if (!simulator || !name || precision != CC3D_FIELD_FLOAT32 || padding != 0) return nullptr;
        try { return simulator->createVectorField(name); } catch (...) { return nullptr; }
    }

    CC3DFieldDimV1 cc3d_shared_vector_dim(void *, CC3DSharedFieldHandle handle, uint8_t) {
        auto *field = static_cast<CompuCell3D::Simulator::vectorField3DNumpyImpl_t *>(handle);
        if (!field) return {0, 0, 0};
        auto dim = field->getDim();
        return {dim.x, dim.y, dim.z};
    }

    uint8_t cc3d_shared_vector_get(void *, CC3DSharedFieldHandle handle, int x, int y, int z, double *vx, double *vy, double *vz, uint8_t) {
        auto *field = static_cast<CompuCell3D::Simulator::vectorField3DNumpyImpl_t *>(handle);
        if (!field || !vx || !vy || !vz) return 0u;
        auto value = field->get(CompuCell3D::Point3D(x, y, z));
        *vx = value.x; *vy = value.y; *vz = value.z;
        return 1u;
    }

    uint8_t cc3d_shared_vector_set(void *, CC3DSharedFieldHandle handle, int x, int y, int z, double vx, double vy, double vz, uint8_t) {
        auto *field = static_cast<CompuCell3D::Simulator::vectorField3DNumpyImpl_t *>(handle);
        if (!field) return 0u;
        field->set(CompuCell3D::Point3D(x, y, z), Coordinates3D<float>(static_cast<float>(vx), static_cast<float>(vy), static_cast<float>(vz)));
        return 1u;
    }

    CC3DFieldDimV1 cc3d_scalar_fields_dim(void *, CC3DFieldHandle fieldHandle) {
        CC3DFieldDimV1 dim{0, 0, 0};
        auto *field = static_cast<CompuCell3D::Field3D<float> *>(fieldHandle);
        if (!field) {
            return dim;
        }

        Dim3D nativeDim = field->getDim();
        dim.x = nativeDim.x;
        dim.y = nativeDim.y;
        dim.z = nativeDim.z;
        return dim;
    }

    float cc3d_scalar_fields_get(void *, CC3DFieldHandle fieldHandle, int x, int y, int z) {
        auto *field = static_cast<CompuCell3D::Field3D<float> *>(fieldHandle);
        if (!field) {
            return 0.0f;
        }
        return field->get(cc3d_point3d(x, y, z));
    }

    void cc3d_scalar_fields_set(void *, CC3DFieldHandle fieldHandle, int x, int y, int z, float value) {
        auto *field = static_cast<CompuCell3D::Field3D<float> *>(fieldHandle);
        if (!field) {
            return;
        }
        field->set(cc3d_point3d(x, y, z), value);
    }

    CC3DFieldDimV1 cc3d_cell_field_dim(void *simulatorPtr) {
        CC3DFieldDimV1 dim{0, 0, 0};
        auto *field = cc3d_cell_field_from_simulator(simulatorPtr);
        if (!field) {
            return dim;
        }

        Dim3D nativeDim = field->getDim();
        dim.x = nativeDim.x;
        dim.y = nativeDim.y;
        dim.z = nativeDim.z;
        return dim;
    }

    CC3DOptionalCellViewV1 cc3d_cell_field_get(void *simulatorPtr, int x, int y, int z) {
        CC3DOptionalCellViewV1 result{};
        auto *field = cc3d_cell_field_from_simulator(simulatorPtr);
        if (!field) {
            return result;
        }

        CompuCell3D::CellG *cell = field->get(cc3d_point3d(x, y, z));
        result.hasCell = cell ? 1u : 0u;
        result.cell = cc3d_cell_view_from_cell(cell);
        return result;
    }

    void cc3d_validate_box_range(const CC3DFieldDimV1 &dim,
                                 int xMin, int xMax,
                                 int yMin, int yMax,
                                 int zMin, int zMax) {
        if (xMin < 0 || yMin < 0 || zMin < 0 ||
            xMin > xMax || yMin > yMax || zMin > zMax ||
            xMax > dim.x || yMax > dim.y || zMax > dim.z) {
            throw CC3DException("Compiled steppable cellField.assign() box is out of range");
        }
    }

    void cc3d_cell_field_assign(void *simulatorPtr,
                                int xMin, int xMax,
                                int yMin, int yMax,
                                int zMin, int zMax,
                                CC3DCellHandle cellHandle) {
        auto *field = cc3d_cell_field_from_simulator(simulatorPtr);
        auto *volumeTracker = cc3d_ensure_volume_tracker(simulatorPtr);
        auto *cell = static_cast<CompuCell3D::CellG *>(cellHandle);

        if (!field) {
            throw CC3DException("Compiled steppable cell field is not initialized");
        }
        if (!volumeTracker) {
            throw CC3DException("Compiled steppable could not initialize VolumeTracker");
        }
        if (!cell) {
            throw CC3DException("Compiled steppable cellField.assign() received a null cell");
        }

        const CC3DFieldDimV1 dim = cc3d_cell_field_dim(simulatorPtr);
        cc3d_validate_box_range(dim, xMin, xMax, yMin, yMax, zMin, zMax);

        for (int z = zMin; z < zMax; ++z) {
            for (int y = yMin; y < yMax; ++y) {
                for (int x = xMin; x < xMax; ++x) {
                    field->set(cc3d_point3d(x, y, z), cell);
                    volumeTracker->step();
                }
            }
        }
    }

    std::vector<CompuCell3D::Point3D> cc3d_collect_cell_pixels(
        void *simulatorPtr, CC3DCellHandle cellHandle
    ) {
        std::vector<CompuCell3D::Point3D> pixels;
        auto iterator = cc3d_cell_pixels_begin(simulatorPtr, cellHandle);
        while (cc3d_cell_pixels_valid(nullptr, iterator)) {
            const CC3DPixelViewV1 pixel = cc3d_cell_pixels_get(nullptr, iterator);
            pixels.emplace_back(static_cast<short>(pixel.x), static_cast<short>(pixel.y), static_cast<short>(pixel.z));
            cc3d_cell_pixels_next(nullptr, iterator);
        }
        cc3d_cell_pixels_destroy(nullptr, iterator);
        return pixels;
    }

    bool cc3d_cell_point_in_lattice(
        const CC3DFieldDimV1 &dim, const CompuCell3D::Point3D &point
    ) {
        return point.x >= 0 && point.x < dim.x &&
               point.y >= 0 && point.y < dim.y &&
               point.z >= 0 && point.z < dim.z;
    }

    bool cc3d_set_cell_pixel(
        void *simulatorPtr, const CompuCell3D::Point3D &point, CC3DCellHandle cellHandle
    ) {
        auto *field = cc3d_cell_field_from_simulator(simulatorPtr);
        auto *volumeTracker = cc3d_ensure_volume_tracker(simulatorPtr);
        if (!field || !volumeTracker) return false;

        field->set(point, static_cast<CompuCell3D::CellG *>(cellHandle));
        volumeTracker->step();
        return true;
    }

    uint8_t cc3d_cell_manipulation_delete(void *simulatorPtr, CC3DCellHandle cellHandle) {
        if (!cellHandle) return 0u;
        const auto pixels = cc3d_collect_cell_pixels(simulatorPtr, cellHandle);
        for (const auto &pixel : pixels) {
            if (!cc3d_set_cell_pixel(simulatorPtr, pixel, nullptr)) return 0u;
        }
        return 1u;
    }

    uint8_t cc3d_cell_manipulation_move(
        void *simulatorPtr, CC3DCellHandle cellHandle, int dx, int dy, int dz
    ) {
        if (!cellHandle) return 0u;
        const auto pixels = cc3d_collect_cell_pixels(simulatorPtr, cellHandle);
        const CC3DFieldDimV1 dim = cc3d_cell_field_dim(simulatorPtr);
        std::set<CompuCell3D::Point3D> oldPixels(pixels.begin(), pixels.end());
        std::set<CompuCell3D::Point3D> newPixels;

        for (const auto &pixel : pixels) {
            const CompuCell3D::Point3D shifted(
                static_cast<short>(pixel.x + dx),
                static_cast<short>(pixel.y + dy),
                static_cast<short>(pixel.z + dz));
            if (cc3d_cell_point_in_lattice(dim, shifted)) newPixels.insert(shifted);
        }

        for (const auto &pixel : newPixels) {
            if (!cc3d_set_cell_pixel(simulatorPtr, pixel, cellHandle)) return 0u;
        }
        for (const auto &pixel : oldPixels) {
            if (newPixels.find(pixel) == newPixels.end() &&
                !cc3d_set_cell_pixel(simulatorPtr, pixel, nullptr)) return 0u;
        }
        return 1u;
    }

    uint8_t cc3d_cell_manipulation_merge(
        void *simulatorPtr, CC3DCellHandle sourceHandle, CC3DCellHandle destinationHandle
    ) {
        if (!sourceHandle || !destinationHandle || sourceHandle == destinationHandle) return 0u;
        const auto pixels = cc3d_collect_cell_pixels(simulatorPtr, sourceHandle);
        for (const auto &pixel : pixels) {
            if (!cc3d_set_cell_pixel(simulatorPtr, pixel, destinationHandle)) return 0u;
        }
        return 1u;
    }

    struct SBMLModelHandleState {
        PyObject *model;
    };

    struct PyGilGuard {
        PyGILState_STATE state;

        PyGilGuard() : state(PyGILState_Ensure()) {}
        ~PyGilGuard() { PyGILState_Release(state); }
    };

    void cc3d_clear_python_error() {
        if (PyErr_Occurred()) {
            PyErr_Clear();
        }
    }

    PyObject *cc3d_unicode_from_cstr(const char *value) {
        if (!value) {
            return nullptr;
        }
        PyObject *pyValue = PyUnicode_FromString(value);
        if (!pyValue) {
            cc3d_clear_python_error();
        }
        return pyValue;
    }

    CC3DSBMLModelHandle cc3d_sbml_find_model(void *, CC3DCellHandle cellHandle, const char *modelName) {
        auto *cell = static_cast<CompuCell3D::CellG *>(cellHandle);
        if (!cell || !cell->pyAttrib || !modelName) {
            return nullptr;
        }

        PyGilGuard gil;
        PyObject *sbmlKey = cc3d_unicode_from_cstr("SBMLSolver");
        PyObject *modelKey = cc3d_unicode_from_cstr(modelName);
        if (!sbmlKey || !modelKey) {
            Py_XDECREF(sbmlKey);
            Py_XDECREF(modelKey);
            return nullptr;
        }

        PyObject *sbmlDict = PyObject_GetItem(cell->pyAttrib, sbmlKey);
        Py_DECREF(sbmlKey);
        if (!sbmlDict) {
            Py_DECREF(modelKey);
            cc3d_clear_python_error();
            return nullptr;
        }

        PyObject *model = PyObject_GetItem(sbmlDict, modelKey);
        Py_DECREF(sbmlDict);
        Py_DECREF(modelKey);
        if (!model) {
            cc3d_clear_python_error();
            return nullptr;
        }

        auto *state = new SBMLModelHandleState{model};
        return static_cast<CC3DSBMLModelHandle>(state);
    }

    void cc3d_sbml_destroy_model(void *, CC3DSBMLModelHandle modelHandle) {
        auto *state = static_cast<SBMLModelHandleState *>(modelHandle);
        if (!state) {
            return;
        }

        PyGilGuard gil;
        Py_XDECREF(state->model);
        delete state;
    }

    PyObject *cc3d_sbml_model_object(CC3DSBMLModelHandle modelHandle) {
        auto *state = static_cast<SBMLModelHandleState *>(modelHandle);
        return state ? state->model : nullptr;
    }

    uint8_t cc3d_sbml_has_value(void *, CC3DSBMLModelHandle modelHandle, const char *valueName) {
        PyObject *model = cc3d_sbml_model_object(modelHandle);
        if (!model || !valueName) {
            return 0u;
        }

        PyGilGuard gil;
        PyObject *key = cc3d_unicode_from_cstr(valueName);
        if (!key) {
            return 0u;
        }
        PyObject *value = PyObject_GetItem(model, key);
        Py_DECREF(key);
        if (!value) {
            cc3d_clear_python_error();
            return 0u;
        }
        Py_DECREF(value);
        return 1u;
    }

    uint8_t cc3d_sbml_get_value(void *, CC3DSBMLModelHandle modelHandle, const char *valueName, double *valueOut) {
        PyObject *model = cc3d_sbml_model_object(modelHandle);
        if (!model || !valueName || !valueOut) {
            return 0u;
        }

        PyGilGuard gil;
        PyObject *key = cc3d_unicode_from_cstr(valueName);
        if (!key) {
            return 0u;
        }
        PyObject *value = PyObject_GetItem(model, key);
        Py_DECREF(key);
        if (!value) {
            cc3d_clear_python_error();
            return 0u;
        }

        double converted = PyFloat_AsDouble(value);
        Py_DECREF(value);
        if (PyErr_Occurred()) {
            cc3d_clear_python_error();
            return 0u;
        }

        *valueOut = converted;
        return 1u;
    }

    uint8_t cc3d_sbml_set_value(void *, CC3DSBMLModelHandle modelHandle, const char *valueName, double value) {
        PyObject *model = cc3d_sbml_model_object(modelHandle);
        if (!model || !valueName) {
            return 0u;
        }

        PyGilGuard gil;
        PyObject *key = cc3d_unicode_from_cstr(valueName);
        PyObject *pyValue = PyFloat_FromDouble(value);
        if (!key || !pyValue) {
            Py_XDECREF(key);
            Py_XDECREF(pyValue);
            cc3d_clear_python_error();
            return 0u;
        }

        int status = PyObject_SetItem(model, key, pyValue);
        Py_DECREF(key);
        Py_DECREF(pyValue);
        if (status != 0) {
            cc3d_clear_python_error();
            return 0u;
        }
        return 1u;
    }

    uint8_t cc3d_sbml_timestep(void *, CC3DSBMLModelHandle modelHandle) {
        PyObject *model = cc3d_sbml_model_object(modelHandle);
        if (!model) {
            return 0u;
        }

        PyGilGuard gil;
        PyObject *result = PyObject_CallMethod(model, "timestep", nullptr);
        if (!result) {
            cc3d_clear_python_error();
            return 0u;
        }
        Py_DECREF(result);
        return 1u;
    }
    CC3DCellViewV1 cc3d_cell_factory_create(void *simulatorPtr, unsigned char cellType) {
        auto *potts = cc3d_potts_from_simulator(simulatorPtr);
        if (!potts) {
            throw CC3DException("Compiled steppable could not access Potts3D to create a cell");
        }

        auto *cell = potts->createCell();
        cell->type = cellType;
        return cc3d_cell_view_from_cell(cell);
    }
}

namespace {
    struct ClusterIteratorState {
        CompuCell3D::CompartmentInventory::compartmentInventoryIterator current;
        CompuCell3D::CompartmentInventory::compartmentInventoryIterator end;
    };

    struct ClusterCellIteratorState {
        CompuCell3D::CC3DCellList cells;
        size_t current = 0;
    };
}

extern "C" CC3D_STEPPABLE_EXPORT CC3DClusterIteratorHandle
cc3d_clusters_begin(void *cellInventoryPtr) {
    auto *inventory = static_cast<CompuCell3D::CellInventory *>(cellInventoryPtr);
    if (!inventory) return nullptr;
    auto &clusters = inventory->getClusterInventory().getContainer();
    return new ClusterIteratorState{clusters.begin(), clusters.end()};
}

extern "C" CC3D_STEPPABLE_EXPORT uint8_t
cc3d_clusters_valid(void *, CC3DClusterIteratorHandle iterator) {
    auto *state = static_cast<ClusterIteratorState *>(iterator);
    return state && state->current != state->end ? 1u : 0u;
}

extern "C" CC3D_STEPPABLE_EXPORT long
cc3d_clusters_get_id(void *, CC3DClusterIteratorHandle iterator) {
    auto *state = static_cast<ClusterIteratorState *>(iterator);
    return state && state->current != state->end ? state->current->first : -1;
}

extern "C" CC3D_STEPPABLE_EXPORT void
cc3d_clusters_next(void *, CC3DClusterIteratorHandle iterator) {
    auto *state = static_cast<ClusterIteratorState *>(iterator);
    if (state && state->current != state->end) ++state->current;
}

extern "C" CC3D_STEPPABLE_EXPORT void
cc3d_clusters_destroy(void *, CC3DClusterIteratorHandle iterator) {
    delete static_cast<ClusterIteratorState *>(iterator);
}

extern "C" CC3D_STEPPABLE_EXPORT CC3DCellIteratorHandle
cc3d_cluster_cells_begin(void *cellInventoryPtr, long clusterId) {
    auto *inventory = static_cast<CompuCell3D::CellInventory *>(cellInventoryPtr);
    if (!inventory) return nullptr;
    auto *state = new ClusterCellIteratorState;
    state->cells = inventory->getClusterCells(clusterId);
    return static_cast<CC3DCellIteratorHandle>(state);
}

extern "C" CC3D_STEPPABLE_EXPORT uint8_t
cc3d_cluster_cells_valid(void *, CC3DCellIteratorHandle iterator) {
    auto *state = static_cast<ClusterCellIteratorState *>(iterator);
    return state && state->current < state->cells.size() ? 1u : 0u;
}

extern "C" CC3D_STEPPABLE_EXPORT CC3DCellViewV1
cc3d_cluster_cells_get(void *, CC3DCellIteratorHandle iterator) {
    auto *state = static_cast<ClusterCellIteratorState *>(iterator);
    if (!state || state->current >= state->cells.size()) return CC3DCellViewV1{};
    return cc3d_cell_view_from_cell(state->cells[state->current]);
}

extern "C" CC3D_STEPPABLE_EXPORT void
cc3d_cluster_cells_next(void *, CC3DCellIteratorHandle iterator) {
    auto *state = static_cast<ClusterCellIteratorState *>(iterator);
    if (state && state->current < state->cells.size()) ++state->current;
}

extern "C" CC3D_STEPPABLE_EXPORT void
cc3d_cluster_cells_destroy(void *, CC3DCellIteratorHandle iterator) {
    delete static_cast<ClusterCellIteratorState *>(iterator);
}

class CompuCell3D::CompiledSteppable::DynamicLibrary {
public:
    explicit DynamicLibrary(const std::string &path) : path_(path), handle_(nullptr) {
#ifdef CC3D_ISWIN
        handle_ = LoadLibrary(path.c_str());
#else
        handle_ = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);
#endif
        if (!handle_) {
            throw CC3DException(buildLoadError("Failed to load compiled steppable library"));
        }
    }

    ~DynamicLibrary() {
        if (!handle_) {
            return;
        }
#ifdef CC3D_ISWIN
        FreeLibrary((HMODULE) handle_);
#else
        dlclose(handle_);
#endif
    }

    void *resolve(const std::string &symbol) const {
#ifdef CC3D_ISWIN
        void *ptr = reinterpret_cast<void *>(GetProcAddress((HMODULE) handle_, symbol.c_str()));
#else
        dlerror();
        void *ptr = dlsym(handle_, symbol.c_str());
#endif
        if (!ptr) {
            throw CC3DException(buildResolveError(symbol));
        }
        return ptr;
    }

private:
    std::string buildLoadError(const std::string &prefix) const {
        std::ostringstream message;
        message << prefix << ": " << path_;
#ifndef CC3D_ISWIN
        const char *error = dlerror();
        if (error) {
            message << " (" << error << ")";
        }
#endif
        return message.str();
    }

    std::string buildResolveError(const std::string &symbol) const {
        std::ostringstream message;
        message << "Compiled steppable entry point '" << symbol << "' not found in " << path_;
#ifndef CC3D_ISWIN
        const char *error = dlerror();
        if (error) {
            message << " (" << error << ")";
        }
#endif
        return message.str();
    }

    std::string path_;
    libHandle_t handle_;
};

CompiledSteppable::CompiledSteppable(const std::string &libraryPath, const std::string &entryPoint)
    : libraryPath_(libraryPath),
      entryPoint_(entryPoint),
      library_(nullptr),
      api_(nullptr),
      context_{},
      state_(nullptr),
      simulator_(nullptr),
      sharedSteppableVars_(nullptr),
      finished_(false),
      initialized_(false) {
    context_.abiVersion = CC3D_KERNEL_ABI_VERSION;
    context_.mcs = 0;
    context_.cellTypes.userdata = nullptr;
    context_.cellTypes.get_fn = &cc3d_cell_types_get;
    context_.cellTypes.name_fn = &cc3d_cell_types_name;
    context_.cells.userdata = nullptr;
    context_.cells.begin_fn = &cc3d_cells_begin;
    context_.cells.valid_fn = &cc3d_cells_valid;
    context_.cells.get_fn = &cc3d_cells_get;
    context_.cells.next_fn = &cc3d_cells_next;
    context_.cells.destroy_fn = &cc3d_cells_destroy;
    context_.cells.by_type_begin_fn = &cc3d_cells_by_type_begin;
    context_.cells.fetch_by_id_fn = &cc3d_cells_fetch_by_id;
    context_.cells.clusterApi.userdata = nullptr;
    context_.cells.clusterApi.clusters_begin_fn = &cc3d_clusters_begin;
    context_.cells.clusterApi.clusters_valid_fn = &cc3d_clusters_valid;
    context_.cells.clusterApi.clusters_get_id_fn = &cc3d_clusters_get_id;
    context_.cells.clusterApi.clusters_next_fn = &cc3d_clusters_next;
    context_.cells.clusterApi.clusters_destroy_fn = &cc3d_clusters_destroy;
    context_.cells.clusterApi.cells_begin_fn = &cc3d_cluster_cells_begin;
    context_.cells.clusterApi.cells_valid_fn = &cc3d_cluster_cells_valid;
    context_.cells.clusterApi.cells_get_fn = &cc3d_cluster_cells_get;
    context_.cells.clusterApi.cells_next_fn = &cc3d_cluster_cells_next;
    context_.cells.clusterApi.cells_destroy_fn = &cc3d_cluster_cells_destroy;
    context_.neighbors.userdata = nullptr;
    context_.neighbors.begin_fn = &cc3d_cell_neighbors_begin;
    context_.neighbors.valid_fn = &cc3d_cell_neighbors_valid;
    context_.neighbors.get_fn = &cc3d_cell_neighbors_get;
    context_.neighbors.next_fn = &cc3d_cell_neighbors_next;
    context_.neighbors.destroy_fn = &cc3d_cell_neighbors_destroy;
    context_.pixels.userdata = nullptr;
    context_.pixels.begin_fn = &cc3d_cell_pixels_begin;
    context_.pixels.valid_fn = &cc3d_cell_pixels_valid;
    context_.pixels.get_fn = &cc3d_cell_pixels_get;
    context_.pixels.next_fn = &cc3d_cell_pixels_next;
    context_.pixels.destroy_fn = &cc3d_cell_pixels_destroy;
    context_.boundaryPixels.userdata = nullptr;
    context_.boundaryPixels.begin_fn = &cc3d_boundary_pixels_begin;
    context_.boundaryPixels.valid_fn = &cc3d_boundary_pixels_valid;
    context_.boundaryPixels.get_fn = &cc3d_boundary_pixels_get;
    context_.boundaryPixels.next_fn = &cc3d_boundary_pixels_next;
    context_.boundaryPixels.destroy_fn = &cc3d_boundary_pixels_destroy;
    context_.scalarFields.userdata = nullptr;
    context_.scalarFields.find_fn = &cc3d_scalar_fields_find;
    context_.scalarFields.dim_fn = &cc3d_scalar_fields_dim;
    context_.scalarFields.get_fn = &cc3d_scalar_fields_get;
    context_.scalarFields.set_fn = &cc3d_scalar_fields_set;
    context_.sharedScalarFields.userdata = nullptr;
    context_.sharedScalarFields.find_fn = &cc3d_shared_scalar_find;
    context_.sharedScalarFields.create_fn = &cc3d_shared_scalar_create;
    context_.sharedScalarFields.dim_fn = &cc3d_shared_scalar_dim;
    context_.sharedScalarFields.get_fn = &cc3d_shared_scalar_get;
    context_.sharedScalarFields.set_fn = &cc3d_shared_scalar_set;
    context_.sharedVectorFields.userdata = nullptr;
    context_.sharedVectorFields.find_fn = &cc3d_shared_vector_find;
    context_.sharedVectorFields.create_fn = &cc3d_shared_vector_create;
    context_.sharedVectorFields.dim_fn = &cc3d_shared_vector_dim;
    context_.sharedVectorFields.get_fn = &cc3d_shared_vector_get;
    context_.sharedVectorFields.set_fn = &cc3d_shared_vector_set;
    context_.cellField.userdata = nullptr;
    context_.cellField.dim_fn = &cc3d_cell_field_dim;
    context_.cellField.get_fn = &cc3d_cell_field_get;
    context_.cellField.assign_fn = &cc3d_cell_field_assign;
    context_.cellManipulation.userdata = nullptr;
    context_.cellManipulation.delete_cell_fn = &cc3d_cell_manipulation_delete;
    context_.cellManipulation.move_cell_fn = &cc3d_cell_manipulation_move;
    context_.cellManipulation.merge_cells_fn = &cc3d_cell_manipulation_merge;
    context_.cellFactory.userdata = nullptr;
    context_.cellFactory.create_fn = &cc3d_cell_factory_create;
    context_.sbml.userdata = nullptr;
    context_.sbml.find_model_fn = &cc3d_sbml_find_model;
    context_.sbml.destroy_model_fn = &cc3d_sbml_destroy_model;
    context_.sbml.has_value_fn = &cc3d_sbml_has_value;
    context_.sbml.get_value_fn = &cc3d_sbml_get_value;
    context_.sbml.set_value_fn = &cc3d_sbml_set_value;
    context_.sbml.timestep_fn = &cc3d_sbml_timestep;
    context_.adhesionFlex.userdata = nullptr;
    context_.adhesionFlex.molecule_count_fn = &cc3d_adhesion_flex_molecule_count;
    context_.adhesionFlex.molecule_name_fn = &cc3d_adhesion_flex_molecule_name;
    context_.adhesionFlex.get_density_fn = &cc3d_adhesion_flex_get_density;
    context_.adhesionFlex.get_density_by_index_fn = &cc3d_adhesion_flex_get_density_by_index;
    context_.adhesionFlex.set_density_fn = &cc3d_adhesion_flex_set_density;
    context_.adhesionFlex.set_density_by_index_fn = &cc3d_adhesion_flex_set_density_by_index;
    context_.adhesionFlex.set_density_vector_fn = &cc3d_adhesion_flex_set_density_vector;
    context_.adhesionFlex.assign_density_vector_fn = &cc3d_adhesion_flex_assign_density_vector;
    context_.adhesionFlex.get_medium_density_fn = &cc3d_adhesion_flex_get_medium_density;
    context_.adhesionFlex.get_medium_density_by_index_fn = &cc3d_adhesion_flex_get_medium_density_by_index;
    context_.adhesionFlex.set_medium_density_fn = &cc3d_adhesion_flex_set_medium_density;
    context_.adhesionFlex.set_medium_density_by_index_fn = &cc3d_adhesion_flex_set_medium_density_by_index;
    context_.adhesionFlex.set_medium_density_vector_fn = &cc3d_adhesion_flex_set_medium_density_vector;
    context_.adhesionFlex.assign_medium_density_vector_fn = &cc3d_adhesion_flex_assign_medium_density_vector;
    context_.lengthConstraint.userdata = nullptr;
    context_.lengthConstraint.set_fn = &cc3d_length_constraint_set;
    context_.lengthConstraint.get_lambda_fn = &cc3d_length_constraint_lambda;
    context_.lengthConstraint.get_target_fn = &cc3d_length_constraint_target;
    context_.lengthConstraint.get_minor_target_fn = &cc3d_length_constraint_minor_target;
    context_.clusterSurface.userdata = nullptr;
    context_.clusterSurface.set_fn = &cc3d_cluster_surface_set;
    context_.clusterSurface.get_fn = &cc3d_cluster_surface_get;
    context_.clusterSurfaceTracker.userdata = nullptr;
    context_.clusterSurfaceTracker.update_cluster_surface_fn = &cc3d_cluster_surface_tracker_update;
    context_.clusterSurfaceTracker.reassign_cluster_id_fn = &cc3d_reassign_cluster_id;
    context_.mitosis.userdata = nullptr;
    context_.mitosis.set_parent_child_position_fn = &cc3d_mitosis_set_position_flag;
    context_.mitosis.get_parent_child_position_fn = &cc3d_mitosis_get_position_flag;
    context_.mitosis.divide_random_fn = &cc3d_mitosis_divide_random;
    context_.mitosis.divide_orientation_fn = &cc3d_mitosis_divide_orientation;
    context_.mitosis.divide_major_axis_fn = &cc3d_mitosis_divide_major;
    context_.mitosis.divide_minor_axis_fn = &cc3d_mitosis_divide_minor;
    context_.mitosis.divide_cluster_random_fn = &cc3d_mitosis_divide_cluster_random;
    context_.mitosis.divide_cluster_orientation_fn = &cc3d_mitosis_divide_cluster_orientation;
    context_.mitosis.divide_cluster_major_axis_fn = &cc3d_mitosis_divide_cluster_major;
    context_.mitosis.divide_cluster_minor_axis_fn = &cc3d_mitosis_divide_cluster_minor;
    context_.mitosis.parent_cell_fn = &cc3d_mitosis_parent;
    context_.mitosis.child_cell_fn = &cc3d_mitosis_child;
    context_.mitosis.clone_attributes_fn = &cc3d_mitosis_clone_attributes;
    context_.orientedGrowth.userdata = nullptr;
    context_.orientedGrowth.set_constraint_width_fn = &cc3d_oriented_growth_set_constraint_width;
    context_.orientedGrowth.set_elongation_axis_fn = &cc3d_oriented_growth_set_axis;
    context_.orientedGrowth.set_elongation_enabled_fn = &cc3d_oriented_growth_set_enabled;
    context_.orientedGrowth.get_elongation_axis_x_fn = &cc3d_oriented_growth_get_axis_x;
    context_.orientedGrowth.get_elongation_axis_y_fn = &cc3d_oriented_growth_get_axis_y;
    context_.orientedGrowth.get_elongation_enabled_fn = &cc3d_oriented_growth_get_enabled;
    context_.momentOfInertia.userdata = nullptr;
    context_.momentOfInertia.get_semiaxes_fn = &cc3d_moment_of_inertia_get_semiaxes;
    context_.momentOfInertia.get_semiaxes_xy_fn = &cc3d_moment_of_inertia_get_semiaxes_xy;
    context_.momentOfInertia.get_semiaxes_xz_fn = &cc3d_moment_of_inertia_get_semiaxes_xz;
    context_.momentOfInertia.get_semiaxes_yz_fn = &cc3d_moment_of_inertia_get_semiaxes_yz;
    context_.momentOfInertia.get_semiaxes_3d_fn = &cc3d_moment_of_inertia_get_semiaxes_3d;
    context_.focalPointPlasticity.userdata = nullptr;
    context_.focalPointPlasticity.create_link_fn = &cc3d_fpp_create;
    context_.focalPointPlasticity.create_internal_link_fn = &cc3d_fpp_create_internal;
    context_.focalPointPlasticity.delete_link_fn = &cc3d_fpp_delete;
    context_.focalPointPlasticity.delete_internal_link_fn = &cc3d_fpp_delete_internal;
    context_.focalPointPlasticity.set_link_fn = &cc3d_fpp_set;
    context_.focalPointPlasticity.get_lambda_fn = &cc3d_fpp_lambda;
    context_.focalPointPlasticity.get_target_fn = &cc3d_fpp_target;
    context_.focalPointPlasticity.create_anchor_fn = &cc3d_fpp_create_anchor;
    context_.focalPointPlasticity.delete_anchor_fn = &cc3d_fpp_delete_anchor;
    context_.focalPointPlasticity.set_anchor_fn = &cc3d_fpp_set_anchor;
    context_.polarizationVector.userdata = nullptr;
    context_.polarizationVector.set_fn = &cc3d_polarization_set;
    context_.polarizationVector.get_fn = &cc3d_polarization_get;
    context_.cellOrientation.userdata = nullptr;
    context_.cellOrientation.set_lambda_fn = &cc3d_cell_orientation_set_lambda;
    context_.cellOrientation.get_lambda_fn = &cc3d_cell_orientation_get_lambda;
    context_.chemotaxis.userdata = nullptr;
    context_.chemotaxis.field_count_fn = &cc3d_chemotaxis_field_count;
    context_.chemotaxis.field_name_fn = &cc3d_chemotaxis_field_name;
    context_.chemotaxis.add_data_fn = &cc3d_chemotaxis_add;
    context_.chemotaxis.get_data_fn = &cc3d_chemotaxis_get;
    context_.secretion.userdata = nullptr;
    context_.secretion.secrete_inside_fn = &cc3d_secrete_inside;
    context_.secretion.secrete_inside_total_fn = &cc3d_secrete_inside_total;
    context_.secretion.secrete_inside_constant_fn = &cc3d_secrete_inside_constant;
    context_.secretion.secrete_inside_constant_total_fn = &cc3d_secrete_inside_constant_total;
    context_.secretion.secrete_inside_boundary_fn = &cc3d_secrete_inside_boundary;
    context_.secretion.secrete_inside_boundary_total_fn = &cc3d_secrete_inside_boundary_total;
    context_.secretion.secrete_inside_boundary_contact_fn = &cc3d_secrete_inside_boundary_contact;
    context_.secretion.secrete_inside_boundary_contact_total_fn = &cc3d_secrete_inside_boundary_contact_total;
    context_.secretion.secrete_outside_boundary_fn = &cc3d_secrete_outside_boundary;
    context_.secretion.secrete_outside_boundary_total_fn = &cc3d_secrete_outside_boundary_total;
    context_.secretion.secrete_outside_boundary_contact_fn = &cc3d_secrete_outside_boundary_contact;
    context_.secretion.secrete_outside_boundary_contact_total_fn = &cc3d_secrete_outside_boundary_contact_total;
    context_.secretion.secrete_com_fn = &cc3d_secrete_com;
    context_.secretion.secrete_com_total_fn = &cc3d_secrete_com_total;
    context_.secretion.uptake_inside_fn = &cc3d_uptake_inside;
    context_.secretion.uptake_inside_total_fn = &cc3d_uptake_inside_total;
    context_.secretion.uptake_boundary_fn = &cc3d_uptake_boundary;
    context_.secretion.uptake_boundary_total_fn = &cc3d_uptake_boundary_total;
    context_.secretion.uptake_outside_boundary_fn = &cc3d_uptake_outside_boundary;
    context_.secretion.uptake_outside_boundary_total_fn = &cc3d_uptake_outside_boundary_total;
    context_.secretion.amount_seen_fn = &cc3d_secretion_amount_seen;
    context_.secretion.field_integral_fn = &cc3d_secretion_field_integral;
    context_.secretion.secrete_single_field_fn = &cc3d_secretion_single;
    context_.secretion.secrete_contact_single_field_fn = &cc3d_secretion_contact_single;
    context_.secretion.secrete_constant_single_field_fn = &cc3d_secretion_constant_single;
    context_.xml.userdata = nullptr;
    context_.xml.get_element_fn = &cc3d_xml_get_element;
    context_.xml.destroy_element_fn = &cc3d_xml_destroy_element;
    context_.xml.get_value_fn = &cc3d_xml_get_value;
    context_.xml.get_value_as_double_fn = &cc3d_xml_get_value_as_double;
    context_.xml.set_value_fn = &cc3d_xml_set_value;
    context_.xml.set_value_as_double_fn = &cc3d_xml_set_value_as_double;
    cc3d_initialize_python_dictionary_api(context_.cellAttributes, nullptr);
    cc3d_initialize_python_dictionary_api(context_.sharedData, nullptr);
}

CompiledSteppable::~CompiledSteppable() {
    try {
        cleanup();
    } catch (...) {
    }
}

void CompiledSteppable::load() {
    ensureLoaded();
    createState();
    finished_ = false;
}

void CompiledSteppable::ensureLoaded() {
    if (api_) {
        return;
    }

    library_.reset(new DynamicLibrary(libraryPath_));

    auto entry = reinterpret_cast<CC3DGetSteppableV1Fn>(library_->resolve(entryPoint_));
    api_ = entry();
    if (!api_) {
        throw CC3DException("Compiled steppable returned a null API table");
    }
    if (api_->abiVersion != CC3D_KERNEL_ABI_VERSION) {
        std::ostringstream message;
        message << "Compiled steppable ABI mismatch: module requires ABI "
                << api_->abiVersion << " but CC3D provides ABI " << CC3D_KERNEL_ABI_VERSION;
        throw CC3DException(message.str());
    }
    if (!api_->step_fn) {
        throw CC3DException("Compiled steppable must provide a step callback");
    }
}

void CompiledSteppable::createState() {
    if (state_ || !api_ || !api_->create_fn) {
        return;
    }

    try {
        state_ = api_->create_fn();
    } catch (const std::exception &e) {
        throw CC3DException(std::string("Compiled steppable create() failed: ") + e.what());
    }
}

void CompiledSteppable::destroyState() {
    if (!state_ || !api_ || !api_->destroy_fn) {
        state_ = nullptr;
        return;
    }

    try {
        api_->destroy_fn(state_);
    } catch (const std::exception &e) {
        throw CC3DException(std::string("Compiled steppable destroy() failed: ") + e.what());
    }
    state_ = nullptr;
}

void CompiledSteppable::attachSimulator(Simulator *simulator) {
    int currentStep = 0;

    if (!simulator) {
        throw CC3DException("CompiledSteppable::attachSimulator received a null simulator");
    }

    ensureLoaded();
    createState();
    finished_ = false;

    simulator_ = simulator;
    context_.cellTypes.userdata = simulator_;
    context_.cells.userdata = &simulator_->getPotts()->getCellInventory();
    context_.cells.clusterApi.userdata = context_.cells.userdata;
    context_.neighbors.userdata = simulator_;
    context_.pixels.userdata = simulator_;
    context_.boundaryPixels.userdata = simulator_;
    context_.scalarFields.userdata = simulator_;
    context_.sharedScalarFields.userdata = simulator_;
    context_.sharedVectorFields.userdata = simulator_;
    context_.cellField.userdata = simulator_;
    context_.cellManipulation.userdata = simulator_;
    context_.cellFactory.userdata = simulator_;
    context_.sbml.userdata = simulator_;
    context_.adhesionFlex.userdata = simulator_;
    context_.lengthConstraint.userdata = simulator_;
    context_.clusterSurface.userdata = simulator_;
    context_.clusterSurfaceTracker.userdata = simulator_;
    context_.mitosis.userdata = simulator_;
    context_.orientedGrowth.userdata = simulator_;
    context_.momentOfInertia.userdata = simulator_;
    context_.focalPointPlasticity.userdata = simulator_;
    context_.polarizationVector.userdata = simulator_;
    context_.cellOrientation.userdata = simulator_;
    context_.chemotaxis.userdata = simulator_;
    context_.secretion.userdata = simulator_;
    context_.xml.userdata = simulator_;
    sharedSteppableVars_ = cc3d_acquire_shared_steppable_vars();
    context_.sharedData.userdata = sharedSteppableVars_;
    currentStep = simulator_->getStep();
    updateContext(currentStep < 0 ? 0u : static_cast<unsigned int>(currentStep));

    if (api_->init_fn && !initialized_) {
        try {
            api_->init_fn(&context_, state_);
            initialized_ = true;
        } catch (const std::exception &e) {
            throw CC3DException(std::string("Compiled steppable init() failed: ") + e.what());
        }
    }
}

void CompiledSteppable::updateContext(unsigned int currentStep) {
    context_.mcs = currentStep;
}

void CompiledSteppable::start() {
    int currentStep = 0;

    if (!simulator_) {
        throw CC3DException("Compiled steppable is not attached to a simulator");
    }

    currentStep = simulator_->getStep();
    updateContext(currentStep < 0 ? 0u : static_cast<unsigned int>(currentStep));
    if (!api_->start_fn) {
        return;
    }

    try {
        api_->start_fn(&context_, state_);
    } catch (const std::exception &e) {
        throw CC3DException(std::string("Compiled steppable start() failed: ") + e.what());
    }
}

void CompiledSteppable::step(unsigned int currentStep) {
    if (!simulator_) {
        throw CC3DException("Compiled steppable is not attached to a simulator");
    }

    updateContext(currentStep);

    try {
        api_->step_fn(&context_, state_);
    } catch (const std::exception &e) {
        throw CC3DException(std::string("Compiled steppable step() failed: ") + e.what());
    }
}

void CompiledSteppable::finish() {
    int currentStep = 0;

    if (finished_) {
        return;
    }

    if (!api_ || !api_->finish_fn || !simulator_) {
        finished_ = true;
        return;
    }

    currentStep = simulator_->getStep();
    updateContext(currentStep < 0 ? 0u : static_cast<unsigned int>(currentStep));

    try {
        api_->finish_fn(&context_, state_);
    } catch (const std::exception &e) {
        throw CC3DException(std::string("Compiled steppable finish() failed: ") + e.what());
    }
    finished_ = true;
}

void CompiledSteppable::cleanup() {
    if (api_ && !finished_ && simulator_) {
        finish();
    }

    destroyState();
    api_ = nullptr;
    library_.reset();
    simulator_ = nullptr;
    initialized_ = false;
    context_.cells.userdata = nullptr;
    context_.cellTypes.userdata = nullptr;
    context_.cells.clusterApi.userdata = nullptr;
    context_.sharedData.userdata = nullptr;
    cc3d_release_python_object(sharedSteppableVars_);
    sharedSteppableVars_ = nullptr;
    context_.neighbors.userdata = nullptr;
    context_.pixels.userdata = nullptr;
    context_.boundaryPixels.userdata = nullptr;
    context_.scalarFields.userdata = nullptr;
    context_.sharedScalarFields.userdata = nullptr;
    context_.sharedVectorFields.userdata = nullptr;
    context_.cellField.userdata = nullptr;
    context_.cellManipulation.userdata = nullptr;
    context_.cellFactory.userdata = nullptr;
    context_.sbml.userdata = nullptr;
    context_.adhesionFlex.userdata = nullptr;
    context_.lengthConstraint.userdata = nullptr;
    context_.clusterSurface.userdata = nullptr;
    context_.clusterSurfaceTracker.userdata = nullptr;
    context_.mitosis.userdata = nullptr;
    context_.orientedGrowth.userdata = nullptr;
    context_.momentOfInertia.userdata = nullptr;
    context_.focalPointPlasticity.userdata = nullptr;
    context_.polarizationVector.userdata = nullptr;
    context_.cellOrientation.userdata = nullptr;
    context_.chemotaxis.userdata = nullptr;
    context_.secretion.userdata = nullptr;
    context_.xml.userdata = nullptr;
}
