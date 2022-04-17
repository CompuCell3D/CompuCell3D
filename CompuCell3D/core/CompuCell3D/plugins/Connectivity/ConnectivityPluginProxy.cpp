

#include "ConnectivityPlugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto connectivityProxy = registerPlugin<Plugin, ConnectivityPlugin>(
    "Connectivity", 
    "Adds connectivity constraints.",
    &Simulator::pluginManager
);
