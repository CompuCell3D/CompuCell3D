

#include "FieldManager.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;


auto fieldManagerProxy = registerPlugin<Steppable,FieldManager>(
	"FieldManager", 
	"It allows users to declare in XML variable precision, numpy fields that are accessible from both Python and C++",
	&Simulator::steppableManager
);

