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

#ifndef STOCHASTICFORCETERM_H
#define STOCHASTICFORCETERM_H



#include "StochasticDLLSpecifier.h"

#include <Components/Interfaces/SingleBodyForceTerm.h>
#include <Components/Interfaces/ModuleApiExporter.h>
#include <Components/CellCM.h>
#include <BasicUtils/BasicRandomNumberGenerator.h>
#include <string>


const char* const moduleName = "Stochastic";
const char* const author = "Maciej Swat";
const char* const moduleType= "SingleBodyForceTerm";
const int versionMajor=3;
const int versionMinor=6;
const int versionSubMinor=2;

namespace CenterModel {

	class SimulationBox;

	class STOCHASTIC_EXPORT StochasticForceTerm: public SingleBodyForceTerm{
    
	public:

		       
		StochasticForceTerm();

		virtual ~StochasticForceTerm();
        
        //ForceTerm interface

        virtual void init(SimulatorCM *_simulator=0, CC3DXMLElement * _xmlData=0);
        virtual Vector3 forceTerm(const CellCM * _cell1);

        virtual std::string getName(){return "Stochastic";}

        //Steerable Interface
        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag=false);
        virtual std::string steerableName(){return getName();}

        
	protected:	
        double mag_min;
        double mag_max;
        double PI;
        double PI_HALF;

        BasicRandomNumberGeneratorNonStatic rGen;
    

    

    };

    MODULE_EXTERNAL_API(STOCHASTIC_EXPORT,SingleBodyForceTerm, StochasticForceTerm)

};
#endif
