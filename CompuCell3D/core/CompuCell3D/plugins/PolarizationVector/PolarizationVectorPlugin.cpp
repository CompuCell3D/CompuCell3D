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

// // // #include <CompuCell3D/Simulator.h>
// // // #include <CompuCell3D/Potts3D/Potts3D.h>
using namespace CompuCell3D;



using namespace std;


#include "PolarizationVectorPlugin.h"

PolarizationVectorPlugin::PolarizationVectorPlugin()   {}

PolarizationVectorPlugin::~PolarizationVectorPlugin() {}

void PolarizationVectorPlugin::setPolarizationVector(CellG * _cell, float _x, float _y, float _z){
   polarizationVectorAccessor.get(_cell->extraAttribPtr)->x=_x;
   polarizationVectorAccessor.get(_cell->extraAttribPtr)->y=_y;
   polarizationVectorAccessor.get(_cell->extraAttribPtr)->z=_z;
}

vector<float> PolarizationVectorPlugin::getPolarizationVector(CellG * _cell){
	vector<float> polarizationVec(3,0.0);
	polarizationVec[0]=polarizationVectorAccessor.get(_cell->extraAttribPtr)->x;
	polarizationVec[1]=polarizationVectorAccessor.get(_cell->extraAttribPtr)->y;
	polarizationVec[2]=polarizationVectorAccessor.get(_cell->extraAttribPtr)->z;
	return polarizationVec;
}

void PolarizationVectorPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
   Potts3D *potts = simulator->getPotts();
   potts->getCellFactoryGroupPtr()->registerClass(&polarizationVectorAccessor); //register new class with the factory
}

void PolarizationVectorPlugin::extraInit(Simulator *simulator) {
}



