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

// // // #define CompuCellLibShared_EXPORTS	// if you dont define this DLL import/export macro  from CompuCellLib you will get error "definition of dllimport static data member not allowed"
									// // // //this is because you define static members in the Simulator class and witohut this macro they will be redefined here as import symbols which is not allowed

// // // #include "Simulator.h"
// // // using namespace CompuCell3D;

// // // #include <BasicUtils/BasicException.h>
// // // #include <BasicUtils/BasicSmartPointer.h>

// // // // #include <XMLCereal/XMLPullParser.h>

// // // // #include <XercesUtils/XercesStr.h>

// // // // #include <xercesc/util/PlatformUtils.hpp>
// // // // XERCES_CPP_NAMESPACE_USE;

// // // #include <iostream>
// // // #include <string>
// // // #include <fstream>
// // // using namespace std;

// // // #include <stdlib.h>

// // // //#include <config.h>
// // // #include <BasicUtils/BasicRandomNumberGenerator.h>

// // // #include <XMLUtils/XMLParserExpat.h>


// // // #if defined(_WIN32)
	// // // #include <windows.h>
// // // #endif


#include <iostream>
#include <string>
#include <Components/CellCM.h>
#include <Components/SimulationBox.h>
#include <Components/CellInventoryCM.h>
#include <Components/CellFactoryCM.h>
#include <time.h>
#include <BasicUtils/BasicRandomNumberGenerator.h>
#include <CompuCell3D/Field3D/Point3D.h>
#include <PublicUtilities/NumericalUtils.h>
#include <limits>
#include <fstream>

#if defined(_WIN32)
	#include <windows.h>
#endif


using namespace std;
using namespace  CenterModel;

// // // PluginManager<Plugin> Simulator::pluginManager;
// // // PluginManager<Steppable> Simulator::steppableManager;
// // // BasicPluginManager<PluginBase> Simulator::pluginBaseManager;


void Syntax(const string name) {
  cerr << "Syntax: " << name << " <config>" << endl;
  exit(1);
}




int main(int argc, char *argv[]) {
	cerr<<"Welcome to CC3D command line edition"<<endl;
    CellCM cell;
    cell.grow();
	cerr<<"cell.position="<<cell.position<<endl;

	Vector3 boxDim(21.2,45.7,80.1);
	//Vector3 gridSpacing(4,10,8);
	Vector3 gridSpacing(2.01,2.01,2.01);
    //Vector3 gridSpacing(3.01,3.01,3.01);
    //Vector3 gridSpacing(1.01,1.01,1.01);

	SimulationBox sb;
	//sb.setDim(21.2,45.7,80.1);
	//sb.setBoxSpatialProperties(21.2,45.7,80.1,1.5,5.5,7.1);
	sb.setBoxSpatialProperties(boxDim,gridSpacing);
	cerr<<sb.getDim()<<endl;

	cerr<<sb.getLatticeLookupDim()<<endl;

	CellFactoryCM cf=CellFactoryCM();
	cf.setSimulationBox(&sb);

    CellInventoryCM ci=CellInventoryCM();
	ci.setCellFactory(&cf);

	//CellCM *cell1=cf.createCellCM(19,20,67);
	//ci.addToInventory(cell1);
	//ci.addToInventory(cf.createCellCM(2,34,21));
	//ci.addToInventory(cf.createCellCM(8,1,9));
	//ci.addToInventory(cf.createCellCM(8,1,10));





	BasicRandomNumberGeneratorNonStatic rGen;
	srand(time(0));
	unsigned int randomSeed=(unsigned int)rand()*((std::numeric_limits<unsigned int>::max)()-1);                
	rGen.setSeed(randomSeed);

	//creating many cells
	int N=20000;
	double r_min=1.0;
	double r_max=2.0;


	
	for (int i=0;i<N;++i){
		CellCM * cell=cf.createCellCM(boxDim.fX*rGen.getRatio(),boxDim.fY*rGen.getRatio(),boxDim.fZ*rGen.getRatio());
		cell->interactionRadius=r_min+rGen.getRatio()*(r_max-r_min);
		ci.addToInventory(cell);
	}




	cerr<<"inventory size="<<ci.getSize()<<endl;

	const SimulationBox::LookupField_t & lookupField=sb.getLookupFieldRef();

	CompuCell3D::Dim3D lookupFieldDim=lookupField.getDim();
	CompuCell3D::Point3D pt;

	int cellCounter=0;

	

#if defined(_WIN32)
	volatile DWORD dwStart;
	dwStart = GetTickCount();
#endif


	Vector3 bc; //boundary condition vector - initialized with (0.,0.,0.)
	
	int n=0;
	double dist=0.0;

    double potential;
    double A=0.2, B=0.1;
    double forceMag;
    Vector3 distVector;

	CellInventoryCM::cellInventoryIterator itr;
	for (itr=ci.cellInventoryBegin() ; itr!=ci.cellInventoryEnd(); ++itr){
		CellCM * cell=itr->second;
        cell->netForce=Vector3(0.,0.,0.);

		InteractionRangeIterator itr = sb.getInteractionRangeIterator(cell);

		itr.begin();
		InteractionRangeIterator endItr = sb.getInteractionRangeIterator(cell).end();

		//cerr<<"***************** NEIGHBORS of cell->id="<<cell->id<<endl;
		CellCM *nCell;
		for (itr.begin(); itr!=endItr ;++itr){
			nCell=(*itr);
			if (nCell==cell)//neglecting "self interactions"
				continue; 
            distVector=distanceVectorInvariantCenterModel(cell->position,nCell->position,boxDim ,bc);
            dist=distVector.Mag();
            potential=A*pow(dist,-12.0)-B*pow(dist,-6.0);
            forceMag=A*(-12.0)*pow(dist,-13.0)-B*(-6.0)*pow(dist,-7.0);
            cell->netForce+=distVector.Unit()*forceMag;
            
            //potential=0.1*pow(dist,6)-0.2*pow(dist,12);
			if (dist<=cell->interactionRadius || dist<=nCell->interactionRadius){

				//cerr<<"**********INTERACTION "<<cell->id<<"-"<<nCell->id<<" ********************"<<endl;
				//cerr<<"THIS IS cell.id="<<nCell->id<<" distance from center cell="<<dist<<endl;
				//cerr<<"nCell->position="<<nCell->position<<" cell->position="<<cell->position<<endl;
    //            cerr<<"nCellLocation="<<sb.getCellLatticeLocation(nCell)<<" cell location="<<sb.getCellLatticeLocation(cell)<<endl;
				++n;
			}

			
		}
		
		//if (n++>20){
		//	break;
		//}
	


	}


	cerr<<"FOUND "<<n<<" interactions of cells"<<endl;
#if defined(_WIN32)
	cerr<<"DISTANCE CALCULATION FOR  "<<N<<" cells too "<<GetTickCount()-dwStart<<" miliseconds to complete"<<endl;
	dwStart = GetTickCount();
#endif	
	
	
	
	
  return 1;
}
