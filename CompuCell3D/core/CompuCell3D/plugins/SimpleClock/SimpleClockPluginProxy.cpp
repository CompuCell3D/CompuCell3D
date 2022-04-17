

#include "SimpleClockPlugin.h"

#include <CompuCell3D/Simulator.h>
using namespace CompuCell3D;

#include <BasicUtils/BasicPluginProxy.h>

BasicPluginProxy<Plugin, SimpleClockPlugin>
simpleClockProxy("SimpleClock", "Simple time(Monte Carlo Step) counter ",
	    &Simulator::pluginManager);
