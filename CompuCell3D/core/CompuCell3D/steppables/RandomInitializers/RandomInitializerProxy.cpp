

#include "RandomFieldInitializer.h"
#include "RandomBlobInitializer.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto RandomFieldInitializerProxy = registerPlugin<Steppable, RandomFieldInitializer>(
        "RandomFieldInitializer",
        "Template Steppable for CompuCell3D Development",
        &Simulator::steppableManager
);

auto RandomBlobInitializerProxy = registerPlugin<Steppable, RandomBlobInitializer>(
        "RandomBlobInitializer",
        "Template Steppable for CompuCell3D Development",
        &Simulator::steppableManager
);
