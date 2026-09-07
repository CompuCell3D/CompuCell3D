#include "CompiledSteppable.h"
#include "cc3d/CompiledSteppablePluginAPI.h"

#include <sstream>
#include <stdexcept>
#include <vector>

#include <Python.h>

#include "CC3DExceptions.h"
#include "Field3D/Field3D.h"
#include "PluginManager.h"
#include "plugins/AdhesionFlex/AdhesionFlexPlugin.h"
#include "plugins/BoundaryPixelTracker/BoundaryPixelTrackerPlugin.h"
#include "plugins/PixelTracker/PixelTrackerPlugin.h"
#include "plugins/NeighborTracker/NeighborTrackerPlugin.h"
#include "plugins/FocalPointPlasticity/FocalPointPlasticityPlugin.h"
#include "plugins/VolumeTracker/VolumeTrackerPlugin.h"
#include "Potts3D/Cell.h"
#include "Potts3D/CellInventory.h"
#include "Potts3D/Potts3D.h"
#include "Simulator.h"

using namespace CompuCell3D;

namespace {
    struct LengthConstraintRegistration { void *simulator; CC3DLengthConstraintCallbacksV1 callbacks; };
    struct AdhesionFlexRegistration { void *simulator; CC3DAdhesionFlexCallbacksV1 callbacks; };
    struct FocalPointPlasticityRegistration { void *simulator; CC3DFocalPointPlasticityCallbacksV1 callbacks; };
    std::vector<LengthConstraintRegistration> lengthConstraintRegistrations;
    std::vector<AdhesionFlexRegistration> adhesionFlexRegistrations;
    std::vector<FocalPointPlasticityRegistration> focalPointPlasticityRegistrations;

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
}

extern "C" void cc3d_register_length_constraint_callbacks(void *simulator, const CC3DLengthConstraintCallbacksV1 *callbacks) {
    if (!simulator || !callbacks) return;
    for (auto &registration : lengthConstraintRegistrations) {
        if (registration.simulator == simulator) { registration.callbacks = *callbacks; return; }
    }
    lengthConstraintRegistrations.push_back({simulator, *callbacks});
}

extern "C" const CC3DLengthConstraintCallbacksV1 *cc3d_get_length_constraint_callbacks(void *simulator) {
    for (const auto &registration : lengthConstraintRegistrations) {
        if (registration.simulator == simulator) return &registration.callbacks;
    }
    return nullptr;
}

extern "C" void cc3d_register_adhesion_flex_callbacks(void *simulator, const CC3DAdhesionFlexCallbacksV1 *callbacks) {
    if (!simulator || !callbacks) return;
    for (auto &r : adhesionFlexRegistrations) if (r.simulator == simulator) { r.callbacks = *callbacks; return; }
    adhesionFlexRegistrations.push_back({simulator, *callbacks});
}
extern "C" const CC3DAdhesionFlexCallbacksV1 *cc3d_get_adhesion_flex_callbacks(void *simulator) {
    for (const auto &r : adhesionFlexRegistrations) if (r.simulator == simulator) return &r.callbacks;
    return nullptr;
}
extern "C" void cc3d_register_focal_point_plasticity_callbacks(void *simulator, const CC3DFocalPointPlasticityCallbacksV1 *callbacks) {
    if (!simulator || !callbacks) return;
    for (auto &r : focalPointPlasticityRegistrations) if (r.simulator == simulator) { r.callbacks = *callbacks; return; }
    focalPointPlasticityRegistrations.push_back({simulator, *callbacks});
}
extern "C" const CC3DFocalPointPlasticityCallbacksV1 *cc3d_get_focal_point_plasticity_callbacks(void *simulator) {
    for (const auto &r : focalPointPlasticityRegistrations) if (r.simulator == simulator) return &r.callbacks;
    return nullptr;
}

namespace {
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

    struct CellIteratorState {
        CompuCell3D::CellInventory::cellInventoryIterator current;
        CompuCell3D::CellInventory::cellInventoryIterator end;
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

    uint8_t cc3d_cells_valid(void *, CC3DCellIteratorHandle iterator) {
        auto *state = static_cast<CellIteratorState *>(iterator);
        return state && state->current != state->end ? 1u : 0u;
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
      finished_(false) {
    context_.abiVersion = CC3D_KERNEL_ABI_VERSION;
    context_.mcs = 0;
    context_.cells.userdata = nullptr;
    context_.cells.begin_fn = &cc3d_cells_begin;
    context_.cells.valid_fn = &cc3d_cells_valid;
    context_.cells.get_fn = &cc3d_cells_get;
    context_.cells.next_fn = &cc3d_cells_next;
    context_.cells.destroy_fn = &cc3d_cells_destroy;
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
    context_.cellField.userdata = nullptr;
    context_.cellField.dim_fn = &cc3d_cell_field_dim;
    context_.cellField.get_fn = &cc3d_cell_field_get;
    context_.cellField.assign_fn = &cc3d_cell_field_assign;
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
    context_.cells.userdata = &simulator_->getPotts()->getCellInventory();
    context_.neighbors.userdata = simulator_;
    context_.pixels.userdata = simulator_;
    context_.boundaryPixels.userdata = simulator_;
    context_.scalarFields.userdata = simulator_;
    context_.cellField.userdata = simulator_;
    context_.cellFactory.userdata = simulator_;
    context_.sbml.userdata = simulator_;
    context_.adhesionFlex.userdata = simulator_;
    context_.lengthConstraint.userdata = simulator_;
    context_.focalPointPlasticity.userdata = simulator_;
    currentStep = simulator_->getStep();
    updateContext(currentStep < 0 ? 0u : static_cast<unsigned int>(currentStep));
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
    context_.cells.userdata = nullptr;
    context_.neighbors.userdata = nullptr;
    context_.pixels.userdata = nullptr;
    context_.boundaryPixels.userdata = nullptr;
    context_.scalarFields.userdata = nullptr;
    context_.cellField.userdata = nullptr;
    context_.cellFactory.userdata = nullptr;
    context_.sbml.userdata = nullptr;
    context_.adhesionFlex.userdata = nullptr;
    context_.lengthConstraint.userdata = nullptr;
    context_.focalPointPlasticity.userdata = nullptr;
}
