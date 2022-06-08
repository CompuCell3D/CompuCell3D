

#include "OrientedContactPlugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto orientedContactProxy = registerPlugin<Plugin, OrientedContactPlugin>(
        "OrientedContact",
        "Adds the interaction energy function and orientation.",
        &Simulator::pluginManager
);
