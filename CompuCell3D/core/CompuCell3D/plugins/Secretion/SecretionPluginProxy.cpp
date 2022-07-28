

#include "SecretionPlugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto secretionProxy = registerPlugin<Plugin, SecretionPlugin>(
        "Secretion",
        "Implements Cellular Secretion",
        &Simulator::pluginManager
);

auto secretionLocalFlexProxy = registerPlugin<Plugin, SecretionPlugin>(
        "SecretionLocalFlex",
        "Implements Cellular Secretion",
        &Simulator::pluginManager
);
