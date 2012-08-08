#include "MolecularContactTrackerPlugin.h"
#include <CompuCell3D/Simulator.h>
using namespace CompuCell3D;

#include <BasicUtils/BasicPluginProxy.h>

BasicPluginProxy<Plugin, MolecularContactTrackerPlugin> 
molecularcontactTrackerPluginProxy("MolecularContactTracker", "Initializes and Tracks MolecularContact participating cells",
	    &Simulator::pluginManager);


