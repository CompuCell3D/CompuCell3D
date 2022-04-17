

#include "TemplatePlugin.h"

#include <CompuCell3D/Simulator.h>
using namespace CompuCell3D;

#include <BasicUtils/BasicPluginProxy.h>

BasicPluginProxy<Plugin, TemplatePlugin> 
templateProxy("TemplatePlugin", "Demonstrates the use of various coding blocks used in CompuCell3D.",
	    &Simulator::pluginManager);

