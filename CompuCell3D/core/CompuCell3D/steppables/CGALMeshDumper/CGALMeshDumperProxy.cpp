
#include "CGALMeshDumper.h"

#include <CompuCell3D/Simulator.h>

using namespace CompuCell3D;

#include <BasicUtils/BasicPluginProxy.h>

BasicPluginProxy <Steppable, CGALMeshDumper>
        cGALMeshDumperProxy("CGALMeshDumper",
                            "Dumps CGAL Mesh",
                            &Simulator::steppableManager);
