

#include "MomentOfInertiaPlugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto momentOfInertiaProxy = registerPlugin<Plugin, MomentOfInertiaPlugin>(
        "MomentOfInertia",
        "Tracks the center of mass for each cell.",
        &Simulator::pluginManager
);
