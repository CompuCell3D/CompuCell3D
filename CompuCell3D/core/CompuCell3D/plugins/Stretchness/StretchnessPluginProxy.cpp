

#include "StretchnessPlugin.h"

#include <CompuCell3D/Simulator.h>
using namespace CompuCell3D;

#include <BasicUtils/BasicPluginProxy.h>

BasicPluginProxy<Plugin, StretchnessPlugin> 
surfaceProxy("Stretchness", "Computes Change in Stretchness Energy",
	    &Simulator::pluginManager);


BasicPluginProxy<Plugin, StretchnessPlugin> 
strechnessEnergyProxy("StretchnessEnergy", "Computes Change in Stretchness Energy",
	    &Simulator::pluginManager);
