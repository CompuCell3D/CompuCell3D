
#include "MitosisPlugin.h"
#include "MitosisSimplePlugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto mitosisProxy = registerPlugin<Plugin, MitosisPlugin>(
        "Mitosis",
        "Splits cells when the reach they doubling volume.",
        &Simulator::pluginManager
);

auto mitosisSimpleProxy = registerPlugin<Plugin, MitosisSimplePlugin>(
        "MitosisSimple",
        "Splits cells when the reach they doubling volume. This version does not register Field Watcher and is intended to be used from Python level",
        &Simulator::pluginManager
);
