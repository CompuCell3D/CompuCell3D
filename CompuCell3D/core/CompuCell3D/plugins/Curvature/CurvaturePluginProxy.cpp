


#include "CurvaturePlugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto curvatureProxy = registerPlugin<Plugin, CurvaturePlugin>(
        "Curvature",
        "computes curvature constraint for 2D compartmental",
        &Simulator::pluginManager
);
