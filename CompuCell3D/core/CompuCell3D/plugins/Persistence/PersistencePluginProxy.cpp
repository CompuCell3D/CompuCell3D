

#include "PersistencePlugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto persistenceProxy = registerPlugin<Plugin, PersistencePlugin>(
        "Persistence",
        "Persistent cell motility models",
        &Simulator::pluginManager
);
