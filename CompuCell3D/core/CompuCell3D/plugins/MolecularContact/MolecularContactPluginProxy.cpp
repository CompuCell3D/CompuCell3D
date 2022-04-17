

#include "MolecularContactPlugin.h"

#include <CompuCell3D/Simulator.h>
using namespace CompuCell3D;

#include <BasicUtils/BasicPluginProxy.h>

BasicPluginProxy<Plugin, MolecularContactPlugin> 
molecularcontactProxy("MolecularContact", "Computes Change in MolecularContact Energy",
	    &Simulator::pluginManager);
