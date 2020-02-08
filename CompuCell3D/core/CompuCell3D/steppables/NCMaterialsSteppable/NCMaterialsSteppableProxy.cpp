#include "NCMaterialsSteppable.h"
#include <CompuCell3D/Simulator.h>

using namespace CompuCell3D;

#include <BasicUtils/BasicPluginProxy.h>

BasicPluginProxy<Steppable, NCMaterialsSteppable>

NCMaterialsSteppableProxy("NCMaterialsSteppable", "Simulates NCMaterial interactions between NCMaterials, fields, and cells",
	    &Simulator::steppableManager);

