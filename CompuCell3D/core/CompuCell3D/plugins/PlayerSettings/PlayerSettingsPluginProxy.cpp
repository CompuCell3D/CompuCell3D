

#include "PlayerSettingsPlugin.h"

#include <CompuCell3D/Simulator.h>
using namespace CompuCell3D;

#include <BasicUtils/BasicPluginProxy.h>

BasicPluginProxy<Plugin, PlayerSettingsPlugin> 
playerSettingsProxy("PlayerSettings", "Keeps some settings for CompuCell Player",
	    &Simulator::pluginManager);
