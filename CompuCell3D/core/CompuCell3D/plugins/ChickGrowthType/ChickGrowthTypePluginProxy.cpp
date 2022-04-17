

#include "ChickGrowthTypePlugin.h"

#include <CompuCell3D/Simulator.h>
using namespace CompuCell3D;

#include <BasicUtils/BasicPluginProxy.h>

BasicPluginProxy<Plugin, ChickGrowthTypePlugin> 
chickGrowthTypeProxy("ChickGrowthType", "Adds cell growth type variable and updates cell types.",
	    &Simulator::pluginManager);
