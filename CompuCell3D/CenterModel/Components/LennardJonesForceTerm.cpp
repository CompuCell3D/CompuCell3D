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

#include "LennardJonesForceTerm.h"
#include <PublicUtilities/NumericalUtils.h>
#include "SimulatorCM.h"
#include <iostream>

using namespace CenterModel;

LennardJonesForceTerm::LennardJonesForceTerm():A(0.2),B(0.1),eps(0.6502),sigma(0.3166){}


LennardJonesForceTerm::~LennardJonesForceTerm(){}

void LennardJonesForceTerm::init(SimulatorCM *_simulator){
    if (!_simulator)
        return;

    simulator=_simulator;
    simulator->registerForce(this);
    
}


//Vector3 LennardJonesForceTerm::forceTerm(const CellCM * _cell1, const CellCM * _cell2, double _distance, const Vector3 & _unitDistVec){
//    Vector3 unitDistVec=_unitDistVec;        
//    double dist=_distance;
//
//    
//    if (!_distance){
//        Vector3 distVec=distanceVectorInvariantCenterModel(_cell1->position , _cell2->position,boxDim ,bc);            
//        dist=distVec.Mag();
//        unitDistVec=distVec.Unit();
//    }
//    double forceMag=A*(-12.0)*pow(dist,-13.0)-B*(-6.0)*pow(dist,-7.0);
//    //cerr<<"forceMag="<<forceMag<<endl;
//    //cerr<<"unitDistVec="<<unitDistVec<<endl;
//
//    return forceMag*unitDistVec;
//    
//}

Vector3 LennardJonesForceTerm::forceTerm(const CellCM * _cell1, const CellCM * _cell2, double _distance, const Vector3 & _unitDistVec){
    Vector3 unitDistVec=_unitDistVec;        
    double dist=_distance;

    
    if (!_distance){
        Vector3 distVec=distanceVectorInvariantCenterModel(_cell1->position , _cell2->position,boxDim ,bc);            
        dist=distVec.Mag();
        unitDistVec=distVec.Unit();
    }
    double forceMag=24.0*(eps/sigma)*(2.0*pow(sigma/dist,13)-pow(sigma/dist,7));
    

    //cerr<<"forceMag="<<forceMag<<endl;
    //cerr<<"unitDistVec="<<unitDistVec<<endl;

    return forceMag*unitDistVec;
    
}

