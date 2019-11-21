#include "ECMaterialsSteppable.h"
#include <CompuCell3D/Simulator.h>

using namespace CompuCell3D;

#include <BasicUtils/BasicPluginProxy.h>

BasicPluginProxy<Steppable, ECMaterialsSteppable>

ECMaterialsSteppableProxy("ECMaterialsSteppable", "Simulates ECMaterial interactions between ECMaterials, fields, and cells",
	    &Simulator::steppableManager);

