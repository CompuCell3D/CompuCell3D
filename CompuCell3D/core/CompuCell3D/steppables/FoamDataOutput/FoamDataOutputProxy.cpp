

#include "FoamDataOutput.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto FoamDataOutputProxy = registerPlugin<Steppable, FoamDataOutput>(
        "FoamDataOutput",
        "Outputs basic simulation data for foam coarsening",
        &Simulator::steppableManager
);
