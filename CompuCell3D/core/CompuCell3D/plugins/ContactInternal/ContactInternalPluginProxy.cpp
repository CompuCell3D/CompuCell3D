

#include "ContactInternalPlugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto contactInternalProxy = registerPlugin<Plugin, ContactInternalPlugin>(
        "ContactInternal",
        "Handles internal adhesion energy between members of the same cluster (i.e. between compartments).",
        &Simulator::pluginManager
);
