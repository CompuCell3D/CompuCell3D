

#include "ConnectivityGlobalPlugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto connectivityGlobalProxy = registerPlugin<Plugin, ConnectivityGlobalPlugin>(
        "ConnectivityGlobal",
        "Adds connectivity constraints imposed globaly using breadth first traversal",
        &Simulator::pluginManager
);
