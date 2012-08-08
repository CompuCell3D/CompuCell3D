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

#ifndef ELASTICITYPLUGIN_H
#define ELASTICITYPLUGIN_H

#include <CompuCell3D/Plugin.h>

#include <CompuCell3D/Potts3D/EnergyFunction.h>
#include <CompuCell3D/Potts3D/Cell.h>

#include "ElasticityDLLSpecifier.h"

class CC3DXMLElement;

namespace CompuCell3D {
  
  
  template <class T> class Field3D;

  class Point3D;
  class Simulator;
  class ElasticityTrackerData;
  class ElasticityTracker;
  class BoundaryStrategy;

  /** 
   * Calculates surface energy based on a target surface and
   * lambda surface.
   */
  class BoundaryStrategy;

  class ELASTICITY_EXPORT ElasticityPlugin : public Plugin, public EnergyFunction{
    
    
    Field3D<CellG *> *cellFieldG;
	std::string pluginName;
	//energy function data
    
    float targetLengthElasticity;
	 float maxLengthElasticity;
    double lambdaElasticity;
    Simulator *simulator;
    Dim3D fieldDim;
    BasicClassAccessor<ElasticityTracker> *elasticityTrackerAccessorPtr;
    typedef double (ElasticityPlugin::*diffEnergyFcnPtr_t)(float _deltaL,float _lBefore,const ElasticityTrackerData * _elasticityTrackerData,const CellG *_cell);

    diffEnergyFcnPtr_t diffEnergyFcnPtr;
    BoundaryStrategy  *boundaryStrategy;


  public:
    ElasticityPlugin();
    virtual ~ElasticityPlugin();

  	 //Plugin interface
    virtual void init(Simulator *_simulator, CC3DXMLElement *_xmlData=0);
	 virtual std::string toString();
	 virtual void extraInit(Simulator *simulator);

	 //EnergyFunction interface
	  virtual double changeEnergy(const Point3D &pt, const CellG *newCell,
                                const CellG *oldCell);


    //SteerableObject interface
    virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag=false);
    virtual std::string steerableName();

	//EnergyFunction methods
    double diffEnergyGlobal(float _deltaL,float _lBefore,const ElasticityTrackerData * _elasticityTrackerData,const CellG *_cell);
    double diffEnergyLocal(float _deltaL,float _lBefore,const ElasticityTrackerData * _elasticityTrackerData,const CellG *_cell);
  


  };
};
#endif
