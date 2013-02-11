

#include "PlasticityTrackerPlugin.h"

#include <CompuCell3D/Simulator.h>
using namespace CompuCell3D;

#include <BasicUtils/BasicPluginProxy.h>

BasicPluginProxy<Plugin, PlasticityTrackerPlugin> 
plasticityTrackerPluginProxy("PlasticityTracker", "Initializes and Tracks Plasticity participating cells",
	    &Simulator::pluginManager);


