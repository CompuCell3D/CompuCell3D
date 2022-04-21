#include "Polarization23Plugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto polarization23Proxy = registerPlugin<Plugin, Polarization23Plugin>(
        "Polarization23",
        "Implements polarization energy term",
        &Simulator::pluginManager
);
