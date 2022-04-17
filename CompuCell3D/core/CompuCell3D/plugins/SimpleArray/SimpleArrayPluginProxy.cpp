

#include <CompuCell3D/plugins/SimpleArray/SimpleArrayPlugin.h>

#include <CompuCell3D/Simulator.h>
#include <BasicUtils/BasicPluginProxy.h>

using namespace CompuCell3D;

BasicPluginProxy<Plugin, SimpleArrayPlugin>
simpleArrayProxy("SimpleArray", "Simple Array template ",
	    &Simulator::pluginManager);
