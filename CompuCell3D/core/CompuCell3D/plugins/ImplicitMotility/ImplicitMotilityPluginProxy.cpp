#include "ImplicitMotilityPlugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto implicitMotilityProxy = registerPlugin<Plugin, ImplicitMotilityPlugin>(
        "ImplicitMotility",
        "Implements implicit motility model",
        &Simulator::pluginManager
);
