#include "BiasVectorSteppable.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto biasVectorSteppableProxy = registerPlugin<Steppable, BiasVectorSteppable>(
        "BiasVectorSteppable",
        "Periodically updates Bias Vector",
        &Simulator::steppableManager
);
