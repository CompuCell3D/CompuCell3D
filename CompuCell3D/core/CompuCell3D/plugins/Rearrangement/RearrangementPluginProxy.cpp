

#include "RearrangementPlugin.h"

#include <CompuCell3D/Simulator.h>
using namespace CompuCell3D;

#include <BasicUtils/BasicPluginProxy.h>

BasicPluginProxy<Plugin, RearrangementPlugin> 
rearrangementEnergyProxy("RearrangementEnergy", "Introduces energy term which suppresses cell rearrangements",
	    &Simulator::pluginManager);

