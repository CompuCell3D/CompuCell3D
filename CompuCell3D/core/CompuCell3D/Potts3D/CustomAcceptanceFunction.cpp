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

#include <CompuCell3D/Simulator.h>
#include "CustomAcceptanceFunction.h"
#include <iostream>
using namespace CompuCell3D;
using namespace std;

double CustomAcceptanceFunction::accept(const double temp, const double change){
	//cerr<<"pUtils="<<pUtils<<endl;
	int currentWorkNodeNumber=pUtils->getCurrentWorkNodeNumber();	
	ExpressionEvaluator & ev=eed[currentWorkNodeNumber];
	double acceptance=0.0;
	//cerr<<"size="<<eed.size()<<endl;
	//cerr<<"temp="<<temp<<endl;
	//cerr<<"change="<<change<<endl;


	ev[0]=temp;
	ev[1]=change;
	
	acceptance=ev.eval();
	//cerr<<"acceptance="<<acceptance<<endl;

	return acceptance;
}

void CustomAcceptanceFunction::initialize(Simulator *_sim){
	if (eed.size()){//this means initialization already happened
		return;
	}
	simulator=_sim;
	pUtils=simulator->getParallelUtils();
	unsigned int maxNumberOfWorkNodes=pUtils->getMaxNumberOfWorkNodesPotts();
	eed.allocateSize(maxNumberOfWorkNodes);
	vector<string> variableNames;

	variableNames.push_back("T");
	variableNames.push_back("DeltaE");	

	eed.addVariables(variableNames.begin(),variableNames.end());


	eed.initializeUsingParseData();

}

void CustomAcceptanceFunction::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){
	eed.getParseData(_xmlData);
	if (_fullInitFlag){
		eed.initializeUsingParseData();
	}
}
