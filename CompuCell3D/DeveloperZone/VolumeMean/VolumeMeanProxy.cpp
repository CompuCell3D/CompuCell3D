

#include "VolumeMean.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>
using namespace CompuCell3D;

auto volumeMeanProxy = registerPlugin<Steppable, VolumeMean>(
        "VolumeMean",
        "Calculates mean volume with user defined exponent",
	    &Simulator::steppableManager
    );

