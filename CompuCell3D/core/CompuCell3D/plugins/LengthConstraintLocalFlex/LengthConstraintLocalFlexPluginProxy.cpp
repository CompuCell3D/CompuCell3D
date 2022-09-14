

#include "LengthConstraintLocalFlexPlugin.h"

#include <CompuCell3D/Simulator.h>
using namespace CompuCell3D;

#include <BasicUtils/BasicPluginProxy.h>

BasicPluginProxy<Plugin, LengthConstraintLocalFlexPlugin> 
lengthConstraintLocalFlexProxy("LengthConstraintLocalFlex", "Tracks cell lengths and adds length constraints. Each individual cell may have different lambda and target lengths",
	    &Simulator::pluginManager);
