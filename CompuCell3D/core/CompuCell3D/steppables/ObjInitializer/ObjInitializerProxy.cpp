

#include "ObjInitializer.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto objInitializerProxy = registerPlugin<Steppable, ObjInitializer>(
        "ObjInitializer",
        "Initializes lattice using user provided OBJ file",
        &Simulator::steppableManager
);
