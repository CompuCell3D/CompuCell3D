

#include "BlobFieldInitializer.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto blobInitializerProxy = registerPlugin<Steppable, BlobFieldInitializer>(
        "BlobInitializer",
        "Initializes lattice by constructing spherical blob of cells",
        &Simulator::steppableManager
);
