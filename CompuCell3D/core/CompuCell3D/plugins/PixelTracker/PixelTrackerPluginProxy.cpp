

#include "PixelTrackerPlugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto pixelTrackerProxy = registerPlugin<Plugin, PixelTrackerPlugin>(
        "PixelTracker",
        "Tracks cell pixels and stores them in the set",
        &Simulator::pluginManager
);
