

#include "CompartmentPlugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto contactProxy = registerPlugin<Plugin, CompartmentPlugin>(
        "ContactCompartment",
        "Adds the interaction energy function.",
        &Simulator::pluginManager
);
