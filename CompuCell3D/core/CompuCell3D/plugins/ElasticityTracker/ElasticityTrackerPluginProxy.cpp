#include "ElasticityTrackerPlugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto elasticityTrackerPluginProxy = registerPlugin<Plugin, ElasticityTrackerPlugin>(
        "ElasticityTracker",
        "Initializes and Tracks Elasticity participating cells",
        &Simulator::pluginManager
);
