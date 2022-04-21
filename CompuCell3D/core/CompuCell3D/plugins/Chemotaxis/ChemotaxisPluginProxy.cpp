

#include "ChemotaxisPlugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto chemotaxisProxy = registerPlugin<Plugin, ChemotaxisPlugin>(
        "Chemotaxis",
        "Adds the chemotactic energy function.",
        &Simulator::pluginManager
);
