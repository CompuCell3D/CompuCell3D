

#include "PIFInitializer.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto pifInitializerProxy = registerPlugin<Steppable, PIFInitializer>(
        "PIFInitializer",
        "Initializes lattice using user provided PIF file",
        &Simulator::steppableManager
);
