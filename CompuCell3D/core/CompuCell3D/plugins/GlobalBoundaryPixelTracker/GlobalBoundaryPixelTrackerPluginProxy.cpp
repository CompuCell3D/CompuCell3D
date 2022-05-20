


#include "GlobalBoundaryPixelTrackerPlugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto globalBoundaryPixelTrackerProxy = registerPlugin<Plugin, GlobalBoundaryPixelTrackerPlugin>(
        "GlobalBoundaryPixelTracker",
        "Tracks  boundary pixels of all the cells including medium and stores them in a set",
        &Simulator::pluginManager
);
