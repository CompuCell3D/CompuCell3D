

#include "ContactLocalProductPlugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto contactLocalProductProxy = registerPlugin<Plugin, ContactLocalProductPlugin>(
        "ContactLocalProduct",
        "Contact energy function. Energy is calculated as a product of cadherins conncentrations",
        &Simulator::pluginManager
);
