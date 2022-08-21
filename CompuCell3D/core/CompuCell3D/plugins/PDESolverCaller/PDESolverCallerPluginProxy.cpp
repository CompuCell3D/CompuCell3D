

#include "PDESolverCallerPlugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto pdeSolverCallerProxy = registerPlugin<Plugin, PDESolverCallerPlugin>(
        "PDESolverCaller",
        "Calls PDE solvers several times during one Monte Carlo Step",
        &Simulator::pluginManager
);
