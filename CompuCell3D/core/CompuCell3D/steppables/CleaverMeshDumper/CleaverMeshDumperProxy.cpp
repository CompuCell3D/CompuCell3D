#include "CleaverMeshDumper.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto cleaverMeshDumperProxy = registerPlugin<Steppable, CleaverMeshDumper>(
        "CleaverMeshDumper",
        "Dumps Cleaver Mesh",
        &Simulator::steppableManager
);
