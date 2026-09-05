#include "CompiledSteppable.h"

#include <sstream>
#include <stdexcept>

#include "CC3DExceptions.h"
#include "Field3D/Field3D.h"
#include "PluginManager.h"
#include "plugins/NeighborTracker/NeighborTrackerPlugin.h"
#include "plugins/VolumeTracker/VolumeTrackerPlugin.h"
#include "Potts3D/Cell.h"
#include "Potts3D/CellInventory.h"
#include "Potts3D/Potts3D.h"
#include "Simulator.h"

using namespace CompuCell3D;

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
    context_.scalarFields.userdata = simulator_;
    context_.cellField.userdata = simulator_;
    context_.cellFactory.userdata = simulator_;
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
    context_.scalarFields.userdata = nullptr;
    context_.cellField.userdata = nullptr;
    context_.cellFactory.userdata = nullptr;
}
