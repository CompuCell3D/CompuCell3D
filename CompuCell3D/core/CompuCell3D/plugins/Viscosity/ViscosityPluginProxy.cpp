

#include "ViscosityPlugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto viscosityProxy = registerPlugin<Plugin, ViscosityPlugin>(
        "Viscosity",
        "Viscosity contact term",
        &Simulator::pluginManager
);
