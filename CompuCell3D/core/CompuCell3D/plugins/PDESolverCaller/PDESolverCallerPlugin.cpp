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

#include <CompuCell3D/CC3D.h>
// // // #include <CompuCell3D/ClassRegistry.h>

using namespace CompuCell3D;


// // // #include <CompuCell3D/Simulator.h>
// // // #include <CompuCell3D/Potts3D/Potts3D.h>


// // // #include <BasicUtils/BasicString.h>
// // // #include <BasicUtils/BasicException.h>

// // // #include <iostream>
using namespace std;

#include "PDESolverCallerPlugin.h"

PDESolverCallerPlugin::PDESolverCallerPlugin():sim(0), potts(0), xmlData(0) {}

PDESolverCallerPlugin::~PDESolverCallerPlugin() {}

void PDESolverCallerPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

	xmlData=_xmlData;
	sim=simulator;
	potts = sim->getPotts();

	potts->registerFixedStepper(this);

	sim->registerSteerableObject(this);

}

void PDESolverCallerPlugin::extraInit(Simulator *simulator) { 

	update(xmlData,true);

}

void PDESolverCallerPlugin::step(){

	//cerr<<" inside STEP"<<endl;
	unsigned int currentStep;
	unsigned int currentAttempt;
	unsigned int numberOfAttempts;


	currentStep=sim->getStep();
	currentAttempt=potts->getCurrentAttempt();
	numberOfAttempts=potts->getNumberOfAttempts();




	for(int i=0 ; i <solverDataVec.size() ; ++i){
		if (! solverDataVec[i].extraTimesPerMC) //when user specifies 0 extra calls per MCS we don't execute the rest of the loop 
			continue;	
		int reminder= (numberOfAttempts % (solverDataVec[i].extraTimesPerMC+1));
		//cerr<<"reminder="<<reminder<<" numberOfAttampts="<<numberOfAttempts<<" solverDataVec[i].extraTimesPerMC="<<solverDataVec[i].extraTimesPerMC<<" currentAttempt="<<currentAttempt<<endl;   
		int ratio=(numberOfAttempts / (solverDataVec[i].extraTimesPerMC+1));
		//       cerr<<"pscpdPtr->solverDataVec[i].extraTimesPerMC="<<pscpdPtr->solverDataVec[i].extraTimesPerMC<<endl;
		//cerr<<"ratio="<<ratio<<" reminder="<<reminder<<endl;
		if( ! ((currentAttempt-reminder) % ratio ) && currentAttempt>reminder ){
			//          cerr<<"before calling step"<<endl;
			solverPtrVec[i]->step(currentStep);
			//          float a=reminder+ratio;
			//cerr<<"calling Solver"<<solverDataVec[i].solverName<<" currentAttempt="<<currentAttempt<<" numberOfAttempts="<<numberOfAttempts<<endl;

		}

	}




}

void PDESolverCallerPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){

	solverPtrVec.clear();
	ClassRegistry *classRegistry=sim->getClassRegistry();
	Steppable * steppable;


	CC3DXMLElementList pdeSolversXMLList=_xmlData->getElements("CallPDE");
	for(unsigned int i=0; i < pdeSolversXMLList.size() ; ++i ){
		solverDataVec.push_back(SolverData(pdeSolversXMLList[i]->getAttribute("PDESolverName"),pdeSolversXMLList[i]->getAttributeAsUInt("ExtraTimesPerMC")));
		SolverData & sd=solverDataVec[solverDataVec.size()-1];

		steppable=classRegistry->getStepper(sd.solverName);
		solverPtrVec.push_back(steppable);

	}


}

std::string PDESolverCallerPlugin::toString(){

	return "PDESolverCaller";

}

std::string PDESolverCallerPlugin::steerableName(){

	return toString();

}
