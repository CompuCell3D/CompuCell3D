

#include "ExternalPotentialPlugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto ExternalPotentialProxy = registerPlugin<Plugin, ExternalPotentialPlugin>(
        "ExternalPotential",
        "Implements external potential energy",
        &Simulator::pluginManager
);
