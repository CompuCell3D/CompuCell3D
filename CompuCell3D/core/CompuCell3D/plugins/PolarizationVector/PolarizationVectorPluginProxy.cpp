

#include "PolarizationVectorPlugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto polarizationVectorProxy = registerPlugin<Plugin, PolarizationVectorPlugin>(
	"PolarizationVector", 
	"Adds polarization vector as a cell attribute",
	&Simulator::pluginManager
);
