

#include "GrowthPlugin.h"

#include <CompuCell3D/Simulator.h>
using namespace CompuCell3D;

#include <BasicUtils/BasicPluginProxy.h>

BasicPluginProxy<Plugin, GrowthPlugin>
growthProxy("Growth", "Adds the growth energy function.", //1,
//             (const char *[]){"Type"},
	     &Simulator::pluginManager);
