

#include "ChickTypePlugin.h"

#include <CompuCell3D/Simulator.h>
using namespace CompuCell3D;

#include <BasicUtils/BasicPluginProxy.h>

BasicPluginProxy<Plugin, ChickTypePlugin> 
chickTypeProxy("ChickType", "Adds cell type variable and updates cell types.",
	    &Simulator::pluginManager);
