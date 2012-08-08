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

#ifndef SIMPLEVOLUMEPLUGIN_H
#define SIMPLEVOLUMEPLUGIN_H

#include <CompuCell3D/Plugin.h>
#include <CompuCell3D/Potts3D/Stepper.h>
#include <CompuCell3D/Potts3D/EnergyFunction.h>
#include <CompuCell3D/Potts3D/CellGChangeWatcher.h>
#include <CompuCell3D/Potts3D/Cell.h>
#include "SimpleVolumeDLLSpecifier.h"
#include <vector>
#include <string>

class CC3DXMLElement;

namespace CompuCell3D {
  class Potts3D;
  class CellG;

  class SIMPLEVOLUME_EXPORT SimpleVolumePlugin : public Plugin , public EnergyFunction 
  {
	Potts3D *potts;
	CC3DXMLElement *xmlData;


  public:
    double targetVolume;
    double lambdaVolume;

    SimpleVolumePlugin():potts(0){};
    virtual ~SimpleVolumePlugin(){};

	//EnergyFunction interface
	virtual double changeEnergy(const Point3D &pt, const CellG *newCell,const CellG *oldCell);

    // SimObject interface
	virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData);
    virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag=false);
    virtual std::string steerableName();
	virtual std::string toString();
  };
};
#endif
