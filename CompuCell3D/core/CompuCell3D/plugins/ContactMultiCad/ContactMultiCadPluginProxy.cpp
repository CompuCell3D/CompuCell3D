

#include "ContactMultiCadPlugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto contactMultiCadProxy = registerPlugin<Plugin, ContactMultiCadPlugin>(
        "ContactMultiCad",
        "Contact energy function. Energy is calculated as a matrix product of cadherins conncentrations",
        &Simulator::pluginManager
);
