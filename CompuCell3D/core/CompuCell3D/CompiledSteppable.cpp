#include "CompiledSteppable.h"

#include <sstream>
#include <stdexcept>

#include "CC3DExceptions.h"
#include "PluginManager.h"
#include "Potts3D/Cell.h"
#include "Potts3D/CellInventory.h"
#include "Simulator.h"

using namespace CompuCell3D;

namespace {
    struct CellIteratorState {
        CompuCell3D::CellInventory::cellInventoryIterator current;
        CompuCell3D::CellInventory::cellInventoryIterator end;
    };

    CC3DCellIteratorHandle cc3d_cells_begin(void *userdata) {
        auto *inventory = static_cast<CompuCell3D::CellInventory *>(userdata);
        auto *state = new CellIteratorState{inventory->cellInventoryBegin(), inventory->cellInventoryEnd()};
        return static_cast<CC3DCellIteratorHandle>(state);
    }

    uint8_t cc3d_cells_valid(void *, CC3DCellIteratorHandle iterator) {
        auto *state = static_cast<CellIteratorState *>(iterator);
        return state && state->current != state->end ? 1u : 0u;
    }

    CC3DCellViewV1 cc3d_cells_get(void *, CC3DCellIteratorHandle iterator) {
        auto *state = static_cast<CellIteratorState *>(iterator);
        CC3DCellViewV1 view{};

        if (!state || state->current == state->end) {
            return view;
        }

        CompuCell3D::CellG *cell = state->current->second;

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

    void cc3d_cells_next(void *, CC3DCellIteratorHandle iterator) {
        auto *state = static_cast<CellIteratorState *>(iterator);
        if (state && state->current != state->end) {
            ++state->current;
        }
    }

    void cc3d_cells_destroy(void *, CC3DCellIteratorHandle iterator) {
        delete static_cast<CellIteratorState *>(iterator);
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
    if (api_->create_fn && !api_->destroy_fn) {
        throw CC3DException("Compiled steppable provides create() but not destroy()");
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
}

