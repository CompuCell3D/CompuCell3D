#include "FocalPointPlasticityPlugin.h"
#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto focalPointPlasticityProxy = registerPlugin<Plugin, FocalPointPlasticityPlugin>(
        "FocalPointPlasticity",
        "allows certain number of cells (user defined) to form plasticity clusters",
        &Simulator::pluginManager
);
