#include "CellOrientationPlugin.h"
#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto CellOrientationProxy = registerPlugin<Plugin, CellOrientationPlugin>(
        "CellOrientation",
        "Computes Change in Cell Orientation Energy",
        &Simulator::pluginManager
);
