

#include "ChemotaxisDictyPlugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto chemotaxisDictyProxy = registerPlugin<Plugin, ChemotaxisDictyPlugin>(
        "ChemotaxisDicty",
        "Adds the chemotactic energy function for dicty.",
        &Simulator::pluginManager
);
