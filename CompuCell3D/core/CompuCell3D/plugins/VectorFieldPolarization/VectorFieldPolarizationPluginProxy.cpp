        

#include "VectorFieldPolarizationPlugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;


auto vectorFieldPolarizationProxy = registerPlugin<Plugin,VectorFieldPolarizationPlugin>(
	"VectorFieldPolarization", 
	"Demonstrates the use of shared numpy vector field",
	&Simulator::pluginManager
);


