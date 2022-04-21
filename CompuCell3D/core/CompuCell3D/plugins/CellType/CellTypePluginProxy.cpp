

#include "CellTypePlugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto cellTypeProxy = registerPlugin<Plugin, CellTypePlugin>(
        "CellType",
        "Adds cell type attributes",
        &Simulator::pluginManager
);
