#ifndef COMPILEDSTEPPABLE_H
#define COMPILEDSTEPPABLE_H

#include <memory>
#include <string>

#include <cc3d/kernel.h>

#include "CompuCellLibDLLSpecifier.h"

namespace CompuCell3D {
    class Simulator;

    class COMPUCELLLIB_EXPORT CompiledSteppable {
    public:
        explicit CompiledSteppable(const std::string &libraryPath,
                                   const std::string &entryPoint = "cc3d_get_steppable_v1");
        ~CompiledSteppable();

        CompiledSteppable(const CompiledSteppable &) = delete;
        CompiledSteppable &operator=(const CompiledSteppable &) = delete;

        void load();
        void attachSimulator(Simulator *simulator);
        void start();
        void step(unsigned int currentStep);
        void finish();
        void cleanup();

        const std::string &getLibraryPath() const { return libraryPath_; }
        const std::string &getEntryPoint() const { return entryPoint_; }

    private:
        class DynamicLibrary;

        void ensureLoaded();
        void createState();
        void destroyState();
        void updateContext(unsigned int currentStep);

        std::string libraryPath_;
        std::string entryPoint_;
        std::unique_ptr<DynamicLibrary> library_;
        const CC3DSteppableV1 *api_;
        CC3DKernelContext context_;
        void *state_;
        Simulator *simulator_;
        bool finished_;
    };
}

#endif
