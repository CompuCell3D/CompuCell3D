

#include "ElasticityPlugin.h"

#include <CompuCell3D/Simulator.h>
using namespace CompuCell3D;

#include <BasicUtils/BasicPluginProxy.h>

BasicPluginProxy<Plugin, ElasticityPlugin> 
elasticityProxy("Elasticity", "Computes Change in Elasticity Energy",
	    &Simulator::pluginManager);


BasicPluginProxy<Plugin, ElasticityPlugin> 
elasticityEnergyProxy("ElasticityEnergy", "Computes Change in Elasticity Energy",
	    &Simulator::pluginManager);
