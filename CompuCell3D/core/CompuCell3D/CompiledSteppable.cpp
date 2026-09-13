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
    struct ClusterSurfaceRegistration { void *simulator; CC3DClusterSurfaceCallbacksV1 callbacks; };
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
    context_.clusterSurface.userdata = nullptr;
    context_.clusterSurface.set_fn = &cc3d_cluster_surface_set;
    context_.clusterSurface.get_fn = &cc3d_cluster_surface_get;
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
    context_.clusterSurface.userdata = simulator_;
    context_.orientedGrowth.userdata = simulator_;
    context_.momentOfInertia.userdata = simulator_;
    context_.focalPointPlasticity.userdata = simulator_;
    context_.polarizationVector.userdata = simulator_;
    context_.cellOrientation.userdata = simulator_;
    context_.chemotaxis.userdata = simulator_;
    context_.secretion.userdata = simulator_;
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
    context_.clusterSurface.userdata = nullptr;
    context_.orientedGrowth.userdata = nullptr;
    context_.momentOfInertia.userdata = nullptr;
    context_.focalPointPlasticity.userdata = nullptr;
    context_.polarizationVector.userdata = nullptr;
    context_.cellOrientation.userdata = nullptr;
    context_.chemotaxis.userdata = nullptr;
    context_.secretion.userdata = nullptr;
}
