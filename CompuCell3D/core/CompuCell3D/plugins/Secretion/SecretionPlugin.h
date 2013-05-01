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


#ifndef SECRETIONPLUGIN_H
#define SECRETIONPLUGIN_H

#include <CompuCell3D/CC3D.h>

// // // #include <CompuCell3D/Plugin.h>
// // // #include <CompuCell3D/Potts3D/FixedStepper.h>
// // // #include <CompuCell3D/Field3D/Dim3D.h>

// // // #include <string>
// // // #include <vector>


#include "SecretionDataP.h"
#include "SecretionDLLSpecifier.h"

class CC3DXMLElement;
namespace CompuCell3D {

  class Potts3D;
  class CellG;
  class Steppable;
  class Simulator;
  class Automaton;
  class BoundaryStrategy;
  class BoxWatcher;
  class BoundaryPixelTrackerPlugin;
  class PixelTrackerPlugin;
  class FieldSecretor;

  class ParallelUtilsOpenMP;

  template <typename Y> class WatchableField3D;
    template <typename Y> class Field3DImpl;
  //class SecretionPlugin;
  


   // class SECRETION_EXPORT SolverData{
      // public:
         // SolverData():extraTimesPerMC(0){}
         // SolverData(std::string _solverName,unsigned int _extraTimesPerMC):
         // solverName(_solverName),
         // extraTimesPerMC(_extraTimesPerMC)
         // {}

         // std::string solverName;
         // unsigned int extraTimesPerMC;

   // };

  //class SECRETION_EXPORT SecretionDataPAdapter : public SecretionDataP {
		//typedef void (SecretionPlugin::*secrSingleFieldFcnPtr_t)(unsigned int idx);
  //};
	//class SECRETION_EXPORT  FieldSecretor{
	//public:

	//	FieldSecretor(){}
	//	~FieldSecretor(){}
	//	Field3DImpl<float> * concentrationFieldPtr;
	//	BoundaryPixelTrackerPlugin *boundaryPixelTrackerPlugin;
	//	PixelTrackerPlugin *pixelTrackerPlugin;

	//	bool secreteInsideCell(CellG * _cell, float _amount);

	//}


  class SECRETION_EXPORT SecretionPlugin : public Plugin, public FixedStepper {
    Potts3D* potts;
    Simulator * sim;
	 CC3DXMLElement *xmlData;
	 
	 // std::vector<SolverData> solverDataVec;

    
    // std::vector<Steppable *> solverPtrVec;
    std::vector<SecretionDataP>  secretionDataPVec;
    Dim3D fieldDim;
	WatchableField3D<CellG *> *cellFieldG;
	Automaton *automaton;
	BoxWatcher *boxWatcherSteppable;
	BoundaryPixelTrackerPlugin *boundaryPixelTrackerPlugin;
	PixelTrackerPlugin *pixelTrackerPlugin;

	ParallelUtilsOpenMP *pUtils;
	BoundaryStrategy *boundaryStrategy;
	unsigned int maxNeighborIndex;
	bool disablePixelTracker;
	bool disableBoundaryPixelTracker;

  public:
    SecretionPlugin();
    virtual ~SecretionPlugin();

	typedef void (SecretionPlugin::*secrSingleFieldFcnPtr_t)(unsigned int idx);

    ///SimObject interface
    virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData=0);
   virtual void  extraInit(Simulator *simulator);

   Field3D<float>*  getConcentrationFieldByName(std::string _fieldName);

   void secreteSingleField(unsigned int idx);
   void secreteOnContactSingleField(unsigned int idx);
   void secreteConstantConcentrationSingleField(unsigned int idx);

   FieldSecretor getFieldSecretor(std::string _fieldName);


    // Stepper interface
    virtual void step();
    

	//bool secreteInsideCell(CellG * _cell, std::string _fieldName, float _amount);
    //SteerableObject interface
    virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag=false);
    virtual std::string steerableName();
    virtual std::string toString();
    
  };
};
#endif

