
#include "ConnectivityLocalFlexPlugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto connectivityLocalFlexProxy = registerPlugin<Plugin, ConnectivityLocalFlexPlugin>(
        "ConnectivityLocalFlex",
        "Adds connectivity constraints based on local parameters",
        &Simulator::pluginManager
);
