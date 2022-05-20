

#include "ConvergentExtensionPlugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto convergentExtensionProxy = registerPlugin<Plugin, ConvergentExtensionPlugin>(
	"ConvergentExtension", 
	"Convergent Extension energy. As described in Zajac , Glazier et al", 
	&Simulator::pluginManager
);
