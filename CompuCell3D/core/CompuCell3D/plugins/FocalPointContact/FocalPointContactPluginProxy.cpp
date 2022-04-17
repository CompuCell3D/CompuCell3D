


#include "FocalPointContactPlugin.h"

#include <CompuCell3D/Simulator.h>
using namespace CompuCell3D;

#include <BasicUtils/BasicPluginProxy.h>


BasicPluginProxy<Plugin, FocalPointContactPlugin>
focalPointContactProxy("FocalPointContact", "implements cell membrane junctions",
	    &Simulator::pluginManager);
