/*************************************************************************
*    CompuCell - A software framework for multimodel simulations of     *
* biocomplexity problems Copyright (C) 2003 University of Notre Dame,   *
*                             Indiana                                   *
*                                                                       *
* This program is free software; IF YOU AGREE TO CITE USE OF CompuCell  *
*  IN ALL RELATED RESEARCH PUBLICATIONS according to the terms of the   *
*  CompuCell GNU General Public License RIDER you can redistribute it   *
* and/or modify it under the terms of the GNU General Public License as *
*  published by the Free Software Foundation; either version 2 of the   *
*         License, or (at your option) any later version.               *
*                                                                       *
* This program is distributed in the hope that it will be useful, but   *
*      WITHOUT ANY WARRANTY; without even the implied warranty of       *
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU    *
*             General Public License for more details.                  *
*                                                                       *
*  You should have received a copy of the GNU General Public License    *
*     along with this program; if not, write to the Free Software       *
*      Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.        *
*************************************************************************/



#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/Potts3D/Potts3D.h>
#include <CompuCell3D/Automaton/Automaton.h>
using namespace CompuCell3D;

#include <iostream>
#include <string>
#include <algorithm>
using namespace std;


#include "SimpleVolumePlugin.h"

void SimpleVolumePlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData)
{
	potts = simulator->getPotts();
	bool pluginAlreadyRegisteredFlag;
	Plugin *plugin=Simulator::pluginManager.get("VolumeTracker",&pluginAlreadyRegisteredFlag); //this will load VolumeTracker plugin if it is not already loaded
	if(!pluginAlreadyRegisteredFlag)
		plugin->init(simulator);
	potts->registerEnergyFunctionWithName(this,toString());
	xmlData=_xmlData;
	
	simulator->registerSteerableObject(this);
	update(_xmlData);
}

void SimpleVolumePlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag)
{
	//if there are no child elements for this plugin it means will use changeEnergyByCellId
	if(_xmlData->findElement("TargetVolume"))
		targetVolume=_xmlData->getFirstElement("TargetVolume")->getDouble();
	if(_xmlData->findElement("LambdaVolume"))
		lambdaVolume=_xmlData->getFirstElement("LambdaVolume")->getDouble();
}

double SimpleVolumePlugin::changeEnergy(const Point3D &pt,const CellG *newCell,const CellG *oldCell) 
{
	/// E = lambda * (volume - targetVolume) ^ 2
	double energy = 0.0;

	if (oldCell == newCell) return 0;

	//as in the original version
	if (newCell){
		energy += lambdaVolume *
			(1 + 2 * (newCell->volume - targetVolume));
	}
	if (oldCell){
		energy += lambdaVolume *
			(1 - 2 * (oldCell->volume - targetVolume));
	}

	return energy;
}

std::string SimpleVolumePlugin::steerableName()
{
	return toString();
}

std::string SimpleVolumePlugin::toString()
{
	return "SimpleVolume";
}

