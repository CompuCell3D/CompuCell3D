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

#ifndef MITOSISPLUGIN_H
#define MITOSISPLUGIN_H
#include <CompuCell3D/CC3D.h>

// // // #include <CompuCell3D/Plugin.h>

// // // #include <CompuCell3D/Potts3D/Cell.h>


// // // #include <CompuCell3D/Potts3D/CellGChangeWatcher.h>
// // // #include <CompuCell3D/Potts3D/Stepper.h>

// // // #include <BasicUtils/BasicArray.h>
// // // #include <vector>
#include "MitosisDLLSpecifier.h"

namespace CompuCell3D {
  class Potts3D;
  class ParallelUtilsOpenMP ;
  class BoundaryStrategy;

  class MITOSIS_EXPORT MitosisPlugin : public Plugin, public CellGChangeWatcher,
			public Stepper {
    protected:

    Potts3D *potts;
	ParallelUtilsOpenMP *pUtils;    
    unsigned int doublingVolume;

    //CellG *childCell;
    //CellG *parentCell;

	std::vector<CellG *> childCellVec;
    std::vector<CellG *> parentCellVec;



    //Point3D splitPt;
    //bool split;
    //bool on;

    std::vector<Point3D> splitPtVec;
    std::vector<short> splitVec; //using shorts instead of bool because vector<bool> has special implementation  not suitable for this plugin
    std::vector<short> onVec;
	std::vector<short> mitosisFlagVec;


    //BasicArray<Point3D> arrays[2];

    unsigned int maxNeighborIndex;
    BoundaryStrategy * boundaryStrategy;

  public:

    MitosisPlugin();
    virtual ~MitosisPlugin();

    // SimObject interface
    virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData=0);
    virtual void handleEvent(CC3DEvent & _event);

    // CellChangeWatcher interface
    virtual void field3DChange(const Point3D &pt, CellG *newCell,
                               CellG *oldCell);

    // Stepper interface
    virtual void step();

    //Steerable interface
    virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag=false);
    virtual std::string steerableName();
    virtual std::string toString();

    // Functions to turn on and off
    virtual void turnOn();
    virtual void turnOff();
    //virtual void turnOnAll();
    //virtual void turnOffAll();

    virtual bool doMitosis();///actually does mitosis - returns true if mitosis was done , false otherwise
    virtual void updateAttributes();///updates some of the cell attributes 
    CellG * getChildCell();
    CellG * getParentCell();

    void setPotts(Potts3D *_potts){potts=_potts;}    

	 unsigned int getDoublingVolume(){return doublingVolume;}
    void setDoublingVolume(unsigned int _doublingVolume){doublingVolume=_doublingVolume;}

   
  };
};
#endif
