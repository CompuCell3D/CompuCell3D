#include "ContactPlugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto contactProxy = registerPlugin<Plugin, ContactPlugin>(
        "Contact",
        "Adds the interaction energy function.",
        &Simulator::pluginManager
);
