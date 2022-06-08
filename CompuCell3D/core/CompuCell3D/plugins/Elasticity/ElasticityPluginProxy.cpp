

#include "ElasticityPlugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto elasticityProxy = registerPlugin<Plugin, ElasticityPlugin>(
	"Elasticity", 
	"Computes Change in Elasticity Energy",
	&Simulator::pluginManager
);

auto elasticityEnergyProxy = registerPlugin<Plugin, ElasticityPlugin>(
	"ElasticityEnergy", "Computes Change in Elasticity Energy",
	&Simulator::pluginManager
);
