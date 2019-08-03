

#include "GrowthSteppable.h"



#include <CompuCell3D/Simulator.h>

using namespace CompuCell3D;



#include <BasicUtils/BasicPluginProxy.h>



BasicPluginProxy<Steppable, GrowthSteppable> 

growthSteppableProxy("GrowthSteppable", "Autogenerated steppeble - the author of the plugin should provide brief description here",

	    &Simulator::steppableManager);        

