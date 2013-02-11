

#include "ElasticityTrackerPlugin.h"

#include <CompuCell3D/Simulator.h>
using namespace CompuCell3D;

#include <BasicUtils/BasicPluginProxy.h>

BasicPluginProxy<Plugin, ElasticityTrackerPlugin> 
elasticityTrackerPluginProxy("ElasticityTracker", "Initializes and Tracks Elasticity participating cells",
	    &Simulator::pluginManager);


