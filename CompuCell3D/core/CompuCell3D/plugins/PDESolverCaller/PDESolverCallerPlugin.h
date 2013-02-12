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


#ifndef PDESOLVERCALLERPLUGIN_H
#define PDESOLVERCALLERPLUGIN_H
#include <CompuCell3D/CC3D.h>
// // // #include <CompuCell3D/Plugin.h>
// // // #include <CompuCell3D/Potts3D/FixedStepper.h>

// // // #include <string>
// // // #include <vector>



#include "PDESolverCallerDLLSpecifier.h"

class CC3DXMLElement;
namespace CompuCell3D {

  class Potts3D;
  class CellG;
  class Steppable;
  class Simulator;


   class PDESOLVERCALLER_EXPORT SolverData{
      public:
         SolverData():extraTimesPerMC(0){}
         SolverData(std::string _solverName,unsigned int _extraTimesPerMC):
         solverName(_solverName),
         extraTimesPerMC(_extraTimesPerMC)
         {}

         std::string solverName;
         unsigned int extraTimesPerMC;

   };

  class PDESOLVERCALLER_EXPORT PDESolverCallerPlugin : public Plugin, public FixedStepper {
    Potts3D* potts;
    Simulator * sim;
	 CC3DXMLElement *xmlData;
	 
	 std::vector<SolverData> solverDataVec;

    
    std::vector<Steppable *> solverPtrVec;
    
  public:
    PDESolverCallerPlugin();
    virtual ~PDESolverCallerPlugin();

    ///SimObject interface
    virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData=0);
   virtual void  extraInit(Simulator *simulator);

    // Stepper interface
    virtual void step();
    


    //SteerableObject interface
    virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag=false);
    virtual std::string steerableName();
    virtual std::string toString();
    
  };
};
#endif

