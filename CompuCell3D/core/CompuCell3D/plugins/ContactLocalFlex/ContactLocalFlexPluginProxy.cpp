#include "ContactLocalFlexPlugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto contactLocalFlexProxy = registerPlugin<Plugin, ContactLocalFlexPlugin>(
        "ContactLocalFlex",
        "Adds the interaction energy function.",
        &Simulator::pluginManager
);
