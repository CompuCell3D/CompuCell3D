

#include "UniformFieldInitializer.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto uniformInitializerProxy = registerPlugin<Steppable, UniformFieldInitializer>(
        "UniformInitializer",
        "Initializes entire lattice with rectangular cells",
        &Simulator::steppableManager
);
