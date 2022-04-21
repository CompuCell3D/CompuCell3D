

#include "PlasticityPlugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto plasticityProxy = registerPlugin<Plugin, PlasticityPlugin>(
        "Plasticity",
        "Computes Change in Plasticity Energy",
        &Simulator::pluginManager
);

auto plasticityEnergyProxy = registerPlugin<Plugin, PlasticityPlugin>(
        "PlasticityEnergy",
        "Computes Change in Plasticity Energy",
        &Simulator::pluginManager
);
