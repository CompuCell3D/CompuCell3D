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

#ifndef SIMULATORCM_H
#define SIMULATORCM_H

#include "ComponentsDLLSpecifier.h"
#include "SimulationBox.h"

#include "CellFactoryCM.h"
#include "CellInventoryCM.h"
#include "ForceCalculator.h"

#include "CellCM.h"

namespace CenterModel {

	class SimulationBox;
    class ForceCalculator;
    class Integrator;

	class COMPONENTS_EXPORT SimulatorCM{
	public:

		SimulatorCM();

		virtual ~SimulatorCM();

        virtual void step();
		
        void init();

        double getCurrentTime(){return timeSim;}
        double getEndTime(){return endTime;}
        double getStartTime(){return startTime;}
        long getCurrentStep(){return stepCounter;}

        Vector3 getBoxDim(){return boxDim;}
        void setBoxDim(double _x, double _y,double _z){boxDim=Vector3(_x,_y,_z);}
        
        Vector3 getGridSpacing(){return gridSpacing;}
        void setGridSpacing(double _x, double _y,double _z){gridSpacing=Vector3(_x,_y,_z);}

        Vector3 getBoundaryConditionVec(){return bc;}
        void setBoundaryConditionVec(double _x, double _y,double _z){bc=Vector3(_x,_y,_z);}

        SimulationBox * getSimulationBoxPtr(){return &sb;}

        CellFactoryCM * getCellFactoryPtr(){return &cf;}
        CellInventoryCM * getCellInventoryPtr(){return &ci;}


        //convenience function - used during testing
        void createRandomCells(int N, double r_min, double r_max,double mot_min, double mot_max);

        void registerForce(ForceTerm * _forceTerm);
        void registerIntegrator(Integrator * _integrator);

	private:

        //SimulationBox *sbPtr;
        SimulationBox sb;
	    CellFactoryCM cf;	    
        CellInventoryCM ci;
        ForceCalculator fCalc;
	    
        Vector3 bc; // boundary condition vector
        Vector3 boxDim; // physical dimensions of computational box
        Vector3 gridSpacing;
        double timeSim;
        double endTime;
        double startTime;
        long stepCounter;

        Integrator * integrator;
	};

};
#endif
