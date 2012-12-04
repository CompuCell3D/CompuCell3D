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

#ifndef SINGLEBODYFORCETERM_H
#define SINGLEBODYFORCETERM_H

#include "InterfacesDLLSpecifier.h"
#include <string>
#include <iostream>

#include <Components/CellCM.h>
#include <Components/Interfaces/CenterModelObject.h>

namespace CenterModel {

	class SimulationBox;
    class SimulatorCM;

	class INTERFACES_EXPORT SingleBodyForceTerm: public CenterModelObject{
    
	public:
		       
		SingleBodyForceTerm();

		virtual ~SingleBodyForceTerm();
        
        //ForceTerm interface
        virtual void init(SimulatorCM *_simulator=0, CC3DXMLElement * _xmlData=0)=0;
        virtual Vector3 forceTerm(const CellCM * _cell1)=0;
		virtual std::string getName()=0;
        

	protected:
        SimulationBox *sbPtr;
        SimulatorCM * simulator;
        Vector3 bc; // boundary condition vector
        Vector3 boxDim; // physical dimensions of computational box
        CC3DXMLElement *xmlData;

	};

};
#endif
