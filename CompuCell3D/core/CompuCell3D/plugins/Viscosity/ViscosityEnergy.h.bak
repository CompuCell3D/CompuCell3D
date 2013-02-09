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

#ifndef VISCOSITYENERGY_H
#define VISCOSITYENERGY_H

#include <CompuCell3D/Potts3D/EnergyFunction.h>

#include <XMLCereal/XMLSerializable.h>

#include <CompuCell3D/Field3D/Point3D.h>
#include <CompuCell3D/Field3D/Dim3D.h>

#include <CompuCell3D/Potts3D/Cell.h>
#include <CompuCell3D/plugins/CellVelocity/InstantVelocityData.h>

#include <map>

template <class T>
class BasicClassAccessor;


namespace CompuCell3D {
  class Potts3D;
  class Automaton;

  class Simulator;
  class CellVelocityData;

  class CenterOfMassPlugin;
  class NeighborTracker;
  class CellInstantVelocityPlugin;
  
  class ViscosityEnergy : public EnergyFunction, public XMLSerializable {
    Potts3D *potts;

//     typedef std::map<int, double> contactEnergies_t;
//     contactEnergies_t contactEnergies;

//     std::string autoName;
    double lambdaViscosity;
    double depth;
    Simulator *simulator;
    Automaton *automaton;
    
  public:
    ViscosityEnergy(Potts3D *potts) :
            potts(potts), lambdaViscosity(0.0), depth(1.0),cellVelocityDataAccessorPtr(0) {}
    
    virtual ~ViscosityEnergy() {}

    virtual double localEnergy(const Point3D &pt);
//      virtual double changeEnergy(const Point3D &pt, const Cell *newCell,
//                                  const Cell *oldCell);

    virtual double changeEnergy(const Point3D &pt, const CellG *newCell,
                                const CellG *oldCell);

    void setSimulator(Simulator * _simulator);
				
    /**
     * @return The contact energy between cell1 and cell2.
     */
//     double contactEnergy(const CellG *cell1, const CellG *cell2);

    /**
     * Sets the contact energy for two cell types.  A -1 type is interpreted
     * as the medium.
     */
//     void setViscosityEnergy(const std::string typeName1,
// 			  const std::string typeName2, const double energy);


    // Begin XMLSerializable interface
    virtual void readXML(XMLPullParser &in);
    virtual void writeXML(XMLSerializer &out);
    // End XMLSerializable interface

    void setCellVelocityDataAccessorPtr(BasicClassAccessor<CellVelocityData> * _ptr);
    void setNeighborTrackerAccessorPtr(BasicClassAccessor<NeighborTracker> * _ptr);
    void setCOMPtr(CenterOfMassPlugin *_ptr);
    void initializeViscosityEnergy();
    
  protected:

   Point3D boundaryConditionIndicator;
   Dim3D fieldDim;
   
   
   BasicClassAccessor<CellVelocityData> * cellVelocityDataAccessorPtr;
   CenterOfMassPlugin * comPluginPtr;
   BasicClassAccessor<NeighborTracker> * neighborTrackerAccessorPtr;
   
//    double dist(double x0,double y0,double z0,double x1,double y1,double z1);
//    double dist(double x0,double y0,double z0);
//    float findMin( float _d , int _dim );

   void precalculateAfterFlipInstantVelocityData(const Point3D &pt, const CellG *newCell, const CellG *oldCell);
   void precalculateAfterFlipCM(const Point3D &pt, const CellG *newCell, const CellG *oldCell);
   
   InstantCellVelocityData ivd;
   CellInstantVelocityPlugin * velPlug;   
      

  };

  inline void ViscosityEnergy::setCellVelocityDataAccessorPtr(BasicClassAccessor<CellVelocityData> *_ptr){
      cellVelocityDataAccessorPtr = _ptr;
  }

  
  inline void ViscosityEnergy::setSimulator(Simulator * _simulator){
      simulator=_simulator;
  }

  inline void ViscosityEnergy::setCOMPtr(CenterOfMassPlugin *_ptr){
      comPluginPtr = _ptr;
  }

  inline void ViscosityEnergy::setNeighborTrackerAccessorPtr(BasicClassAccessor<NeighborTracker> * _ptr){
      neighborTrackerAccessorPtr=_ptr;
  }
  
};
#endif
