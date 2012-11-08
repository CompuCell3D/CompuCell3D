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

	SimulationBox sb;
	sb.setDim(21.2,45.7,80.1);
	sb.setBoxSpatialProperties(21.2,45.7,80.1,1.5,5.5,7.1);
	cerr<<sb.getDim()<<endl;

	cerr<<sb.getLatticeLookupDim()<<endl;

	CellFactoryCM cf=CellFactoryCM();
    CellInventoryCM ci=CellInventoryCM();
	ci.addToInventory(cf.createCellCM());
	ci.addToInventory(cf.createCellCM());
	ci.addToInventory(cf.createCellCM());

	cerr<<"inventory size="<<ci.getSize()<<endl;

	
	



  return 1;
}
