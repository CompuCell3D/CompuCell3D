#include "BoundaryPixelTrackerPlugin.h"
#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto boundaryPixelTrackerProxy = registerPlugin<Plugin, BoundaryPixelTrackerPlugin>(
        "BoundaryPixelTracker",
        "Tracks  cells' boundary pixels and stores them in the set",
        &Simulator::pluginManager
);
