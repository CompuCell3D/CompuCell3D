

#include "TubeFieldInitializer.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;


auto tubeFieldInitializerProxy = registerPlugin<Steppable,TubeFieldInitializer>(
	"TubeInitializer", 
	"Initializes lattice by constructing a tube of a given length and thickness",
	&Simulator::steppableManager
);

