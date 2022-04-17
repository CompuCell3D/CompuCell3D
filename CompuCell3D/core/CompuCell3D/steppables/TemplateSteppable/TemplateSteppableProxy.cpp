

#include "TemplateSteppable.h"

#include <CompuCell3D/Simulator.h>
using namespace CompuCell3D;

#include <BasicUtils/BasicPluginProxy.h>

BasicPluginProxy<Steppable, TemplateSteppable> 
pifInitializerProxy("TemplateSteppable", "Template Steppable for CompuCell3D Development",
	    &Simulator::steppableManager);
