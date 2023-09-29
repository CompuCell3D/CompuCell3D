

#include "PolygonFieldInitializer.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;


auto polygonFieldInitializerProxy = registerPlugin<Steppable,PolygonFieldInitializer>(
	"PolygonInitializer", 
	"Initializes lattice by constructing any arbitrary shape of cells",
	&Simulator::steppableManager
);

