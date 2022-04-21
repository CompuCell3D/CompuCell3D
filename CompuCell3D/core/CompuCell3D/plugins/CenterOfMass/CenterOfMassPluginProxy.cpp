

#include "CenterOfMassPlugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto centerOfMassProxy = registerPlugin<Plugin, CenterOfMassPlugin>(
        "CenterOfMass",
        "Tracks the center of mass for each cell.",
        &Simulator::pluginManager
);
