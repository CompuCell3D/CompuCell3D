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


//#include "GrowthRenderer.h"


#include <CompuCell3D/plugins/Mitosis/MitosisPlugin.h>

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/ClassRegistry.h>
#include <CompuCell3D/Potts3D/Potts3D.h>
#include <CompuCell3D/Field3D/Field3D.h>

#include <XMLCereal/XMLPullParser.h>
#include <XMLCereal/XMLSerializer.h>

#include <BasicUtils/BasicString.h>
#include <BasicUtils/BasicException.h>

#include <iostream>


#include "GrowthPlugin.h"

using namespace std;
using namespace CompuCell3D;

GrowthPlugin::GrowthPlugin() {
  stopaer = -1; // Default is to always have AER, unless told not to
  zAER = 0;
}

GrowthPlugin::~GrowthPlugin() {}

void GrowthPlugin::init(Simulator *simulator) {
  sim = simulator;
  potts = simulator->getPotts(); 

  potts->registerCellGChangeWatcher(this);
  potts->registerStepper(this);
  
  evolution = -1;
  currstep = 0;
  numcellpixels = 0;
  density = 0;
  dTrigger = false;

  mitosisPlugin = ((MitosisPlugin*)Simulator::pluginManager.get("Mitosis"));

  //TypePlugin *typePlugin = (TypePlugin *)Simulator::pluginManager.get("Type");

  //if (!typePlugin) cerr << "WARNING: NULL TYPE PLUGIN!!" << endl;
 // ClassRegistry *reg = simulator->getClassRegistry();

  //reg->registerRenderer("Growth",
//			new GrowthRendererFactory(growthEnergy,
//						    potts->getCellField()));
}

void GrowthPlugin::extraInit(Simulator *simulator) {
  sim = simulator;
  //potts->getCellFieldG()->setDim(dim);
}

double GrowthPlugin::getFGFThreshold() {
	return fgfThreshold;
}

double GrowthPlugin::FGF(int z)
{
   Dim3D dim = potts->getCellFieldG()->getDim();
   // Piecewise linear function for FGF concentration
   if (z >= zAER)
       return (((double)(z-zAER)/(dim.z-zAER))*((double)(1-fgfThreshold)) + fgfThreshold);
   else
       return ((double)(z/zAER))*(fgfThreshold-0.1) + 0.1;
}

void GrowthPlugin::field3DChange(const Point3D &pt, CellG *newCell,
	    		       CellG *oldCell) {

    if (newCell && !oldCell) numcellpixels++;
    else if (oldCell && !newCell) numcellpixels--;
    density = ((double) numcellpixels / 
            (double) (potts->getCellFieldG()->getDim().x *
                      potts->getCellFieldG()->getDim().y *
                      potts->getCellFieldG()->getDim().z)) * 100.0; 
}

void GrowthPlugin::grow()
{
   potts->getCellFieldG()->setDim(Dim3D(potts->getCellFieldG()->getDim().x,
                                       potts->getCellFieldG()->getDim().y,
                                       potts->getCellFieldG()->getDim().z+delta));
   zAER += delta;
}

void GrowthPlugin::step()
{
   if (sim->getStep() == 0) potts->getCellFieldG()->setDim(dim);
   int stepnumber = sim->getStep();
   //if (stepnumber == stopaer) growthEnergy->aerOff();
   if (stepnumber != evolution)
   {       
       evolution = stepnumber;
   if ((density > dThreshold) && (!dTrigger))
   {
      cerr << "***** GROWING *****" << endl;
      grow();
      dTrigger = true;
      mitosisPlugin->turnOff();
      currstep++;
   }
   else if (dTrigger)
   {
       if (currstep == delay)
       {
          dTrigger = false;
          //if (density <= dThreshold)
          mitosisPlugin->turnOn();
          currstep = 0;
       }
       else
           currstep++;
   }
   }
}

void GrowthPlugin::readXML(XMLPullParser &in) {
  in.skip(TEXT);

  while (in.check(START_ELEMENT)) {
    if (in.getName() == "Delta") {
      delta = BasicString::parseUInteger(in.matchSimple());
    } 
    else if (in.getName() == "DensityThreshold") {
        dThreshold = BasicString::parseDouble(in.matchSimple());
        ASSERT_OR_THROW("Threshold must be >= 0 and <= 100!", ((dThreshold >= 0) && (dThreshold <= 100)));
    }
    else if (in.getName() == "Delay") {
        delay = BasicString::parseUInteger(in.matchSimple());
    }
    else if (in.getName() == "FGFThreshold") {
        fgfThreshold = BasicString::parseDouble(in.matchSimple());
        ASSERT_OR_THROW("Threshold must be >= 0 and <= 100!", ((fgfThreshold >= 0) && (fgfThreshold <= 100)));
    }
    else if (in.getName() == "InitialDim") {
        dim.readXML(in);
        in.matchSimple();
    }
    else if (in.getName() == "StopAER") {
      stopaer = BasicString::parseInteger(in.matchSimple());
    }
    else {
      THROW(string("Unexpected element '") + in.getName() + "'!");
    }
    
    in.skip(TEXT);
  }
}

void GrowthPlugin::writeXML(XMLSerializer &out) {
}
