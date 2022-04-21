

#include "PIFDumper.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto pifDumperProxy = registerPlugin<Steppable, PIFDumper>(
        "PIFDumper",
        "Stores lattice as a PIF file",
        &Simulator::steppableManager
);
