

#include "BoxWatcher.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto boxWatcherProxy = registerPlugin<Steppable, BoxWatcher>(
        "BoxWatcher",
        "Monitors and updates dimension of the rectangular box in which non-frozen cells are contained",
        &Simulator::steppableManager
);
