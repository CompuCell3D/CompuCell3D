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
using namespace CompuCell3D;
using namespace std;

#include "PIFInitializer.h"

PIFInitializer::PIFInitializer() :
potts(0),sim(0), pifname("") {}

PIFInitializer::PIFInitializer(string filename) :
potts(0),sim(0), pifname(filename) {}

void PIFInitializer::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
   
   sim=simulator;

   bool pluginAlreadyRegisteredFlag;
   Plugin *plugin=Simulator::pluginManager.get("VolumeTracker",&pluginAlreadyRegisteredFlag); //this will load VolumeTracker plugin if it is not already loaded
  if(!pluginAlreadyRegisteredFlag)
      plugin->init(simulator);

	pifname=_xmlData->getFirstElement("PIFName")->getText();

	std::string basePath=simulator->getBasePath();
	cerr << "basePath=simulator->getBasePath()=" << simulator->getBasePath() << endl;
	if (basePath!=""){
		pifname	= basePath+"/"+pifname;
	}


	potts = simulator->getPotts();
}

void PIFInitializer::start() {
	if (sim->getRestartEnabled()){
		return ;  // we will not initialize cells if restart flag is on
	}

	cerr<<"ppdPtr->pifname="<<pifname<<endl;

	std::ifstream piffile(pifname.c_str(), ios::in);
	cerr<<"opened pid file"<<endl;
	ASSERT_OR_THROW(string("Could not open\n"+pifname+"\nMake sure it exists and is in correct directory"),piffile.good());
	WatchableField3D<CellG *> * cellFieldG =(WatchableField3D<CellG *> *) potts->getCellFieldG();
	ASSERT_OR_THROW("initField() Cell field cannot be null!", cellFieldG);

	Dim3D dim = cellFieldG->getDim();
	cerr<<"THIS IS DIM FOR PIF "<<dim<<endl;

	long spin;
	long clusterId;
	std::string celltype;
	std::string first;
	std::string second;
	std::string line;

	int xLow, xHigh, yLow, yHigh, zLow, zHigh;
	std::map<long, Point3D> spinMap; // Used to check if a cell of the same spin is
	// listed twice.

	Point3D cellPt;
	CellG* cell;

	TypeTransition * typeTransitionPtr=potts->getTypeTransition();

	cerr<<"typeTransitionPtr="<<typeTransitionPtr<<endl;
	getline(piffile,line);
	istringstream pif(line);
	pif >> first >> second;
	cerr << "First: " << first << " Second: " << second << "\n";
	if(second == "Clusters") {
		cerr << "Clusters Included" << "\n";
		while(getline(piffile,line)) {
			istringstream pif(line);
			pif >> clusterId>> spin >> celltype >> xLow;

			//             cerr << "  Cluster Id:  " <<clusterId<< "  Spin: " << spin 
			//                << "  Type: " <<  celltype <<"\n";

			ASSERT_OR_THROW(string("PIF reader: xLow out of bounds : \n")+line, xLow >= 0 && xLow < dim.x);
			pif >> xHigh;
			ASSERT_OR_THROW(string("PIF reader: xHigh out of bounds : \n") + line, xHigh >= 0 && xHigh < dim.x);
			ASSERT_OR_THROW(string("PIF reader: xHigh is smaller than xLow : \n") + line, xHigh >= xLow); 
			pif >> yLow;
			ASSERT_OR_THROW(string("PIF reader: yLow out of bounds : \n") + line, yLow >= 0 && yLow < dim.y);
			pif >> yHigh;   
			ASSERT_OR_THROW(string("PIF reader: yHigh out of bounds : \n") + line, yHigh >= 0 && yHigh < dim.y);
			ASSERT_OR_THROW(string("PIF reader: yHigh is smaller than yLow : \n") + line, yHigh >= yLow);
			pif >> zLow;
			ASSERT_OR_THROW(string("PIF reader: zLow out of bounds : \n") + line, zLow >= 0 && zLow < dim.z);
			pif >> zHigh;
			ASSERT_OR_THROW(string("PIF reader: zHigh out of bounds : \n") + line, zHigh >= 0 && zHigh < dim.z);
			ASSERT_OR_THROW(string("PIF reader: zHigh is smaller than xLow : \n") + line, zHigh >= zLow);
			if (spinMap.count(spin) != 0) // Spin multiply listed
			{
				for (cellPt.z = zLow; cellPt.z <= zHigh; cellPt.z++)
					for (cellPt.y = yLow; cellPt.y <= yHigh; cellPt.y++)
						for (cellPt.x = xLow; cellPt.x <= xHigh; cellPt.x++){
							cellFieldG->set(cellPt, cellFieldG->get(spinMap[spin]));
							potts->runSteppers(); //used to ensure that VolumeTracker Plugin step fcn gets called every time we do something to the fields
							//It is necessary to do it this way because steppers are called only when we are performing pixel copies
							// but if we initialize steppers are not called thus is you overwrite a cell here it will not get removed from
							//inventory unless you call steppers(VolumeTrackerPlugin) explicitely

						}


			}
			else // First time for this spin, we need to create a new cell
			{
				spinMap[spin] = Point3D(xLow, yLow, zLow);
				cell = potts->createCellGSpecifiedIds(Point3D(xLow, yLow, zLow),spin,clusterId);
				
				cell->type=potts->getAutomaton()->getTypeId(celltype);//first manually set cell type , then we reset it  via setType of transition Ptr  (transitionPtr is obsolete and not really used in most recent CC3D versions)

				potts->runSteppers(); //used to ensure that VolumeTracker Plugin step fcn gets called every time we do something to the fields
				//It is necessary to do it this way because steppers are called only when we are performing pixel copies
				// but if we initialize steppers are not called thus is you overwrite a cell here it will not get removed from
				//inventory unless you call steppers(VolumeTrackerPlugin) explicitely

				for (cellPt.z = zLow; cellPt.z <= zHigh; cellPt.z++)
					for (cellPt.y = yLow; cellPt.y <= yHigh; cellPt.y++)
						for (cellPt.x = xLow; cellPt.x <= xHigh; cellPt.x++){
							cellFieldG->set(cellPt, cell);
							potts->runSteppers(); //used to ensure that VolumeTracker Plugin step fcn gets called every time we do something to the fields
							//It is necessary to do it this way because steppers are called only when we are performing pixel copies
							// but if we initialize steppers are not called thus is you overwrite a cell here it will not get removed from
							//inventory unless you call steppers(VolumeTrackerPlugin) explicitely

						}

						//         cell->type=potts->getAutomaton()->getTypeId(celltype);             

						//cerr << "CELLTYPE: " << celltype << "\n";
						//cerr << "CLUSTERID: " << clusterId << "\n";
						typeTransitionPtr->setType( cell,potts->getAutomaton()->getTypeId(celltype));
						//cell->clusterId=clusterId;
						//cerr << "1. Cell Type from cell->:  " <<(int)cell->type<< "\n";
						//cerr << "1. ClusterID from cell->:  " <<(int)cell->clusterId<< "\n";


			}


		}
	}
	else {
		//cerr <<"\n\n\n Only Cell Types" << "\n";
		pif >> xLow;
		int tmp = atoi(first.c_str());
		spin = tmp;
		celltype = second;
		//cerr << "spin: " << spin << " celltype: : " << celltype << 
		//     " xLow: " << xLow << endl;
		ASSERT_OR_THROW(string("PIF reader: xLow out of bounds : \n") + line, xLow >= 0 && xLow < dim.x);
		pif >> xHigh;
		ASSERT_OR_THROW(string("PIF reader: xHigh out of bounds : \n")+line, xHigh >= 0 && xHigh < dim.x);
		ASSERT_OR_THROW(string("PIF reader: xHigh is smaller than xLow : \n")+line, xHigh >= xLow); 
		pif >> yLow;
		ASSERT_OR_THROW(string("PIF reader: yLow out of bounds : \n")+line, yLow >= 0 && yLow < dim.y);
		pif >> yHigh;   
		ASSERT_OR_THROW(string("PIF reader: yHigh out of bounds : \n")+line, yHigh >= 0 && yHigh < dim.y);
		ASSERT_OR_THROW(string("PIF reader: yHigh is smaller than yLow : \n")+line, yHigh >= yLow);
		pif >> zLow;
		ASSERT_OR_THROW(string("PIF reader: zLow out of bounds : \n")+line, zLow >= 0 && zLow < dim.z);
		pif >> zHigh;
		ASSERT_OR_THROW(string("PIF reader: zHigh out of bounds : \n")+line, zHigh >= 0 && zHigh < dim.z);
		ASSERT_OR_THROW(string("PIF reader: zHigh is smaller than xLow : \n")+line, zHigh >= zLow);

		if (spinMap.count(spin) != 0) // Spin multiply listed
		{
			for (cellPt.z = zLow; cellPt.z <= zHigh; cellPt.z++)
				for (cellPt.y = yLow; cellPt.y <= yHigh; cellPt.y++)
					for (cellPt.x = xLow; cellPt.x <= xHigh; cellPt.x++){
						cellFieldG->set(cellPt, cellFieldG->get(spinMap[spin]));
						potts->runSteppers(); //used to ensure that VolumeTracker Plugin step fcn gets called every time we do something to the fields
						//It is necessary to do it this way because steppers are called only when we are performing pixel copies
						// but if we initialize steppers are not called thus is you overwrite a cell here it will not get removed from
						//inventory unless you call steppers(VolumeTrackerPlugin) explicitely

					}
		}
		else // First time for this spin, we need to create a new cell
		{
			spinMap[spin] = Point3D(xLow, yLow, zLow);
			cell = potts->createCellGSpecifiedIds(Point3D(xLow, yLow, zLow),spin);
			cell->type=potts->getAutomaton()->getTypeId(celltype);//first manually set cell type , then we reset it  via setType of transition Ptr  (transitionPtr is obsolete and not really used in most recent CC3D versions)

			potts->runSteppers(); //used to ensure that VolumeTracker Plugin step fcn gets called every time we do something to the fields
			//It is necessary to do it this way because steppers are called only when we are performing pixel copies
			// but if we initialize steppers are not called thus is you overwrite a cell here it will not get removed from
			//inventory unless you call steppers(VolumeTrackerPlugin) explicitely

			for (cellPt.z = zLow; cellPt.z <= zHigh; cellPt.z++)
				for (cellPt.y = yLow; cellPt.y <= yHigh; cellPt.y++)
					for (cellPt.x = xLow; cellPt.x <= xHigh; cellPt.x++){
						cellFieldG->set(cellPt, cell);
						potts->runSteppers(); //used to ensure that VolumeTracker Plugin step fcn gets called every time we do something to the fields
						//It is necessary to do it this way because steppers are called only when we are performing pixel copies
						// but if we initialize steppers are not called thus is you overwrite a cell here it will not get removed from
						//inventory unless you call steppers(VolumeTrackerPlugin) explicitely

					}


					typeTransitionPtr->setType(cell , potts->getAutomaton()->getTypeId(celltype));
					//cerr << "1. Cell Type from cell->:  " <<(int)cell->type<< "\n";
					// cerr << "getline(pif,line): " << getline(pif,line) << endl;
					//    cerr << "getline(piffline,line): " <<  getline(piffile,line) << endl;
		}
        while(getline(piffile,line) ) {
			//cerr << "PINGPINGPINGPINGP5Ng" << endl;
			istringstream pif(line);
			pif >> spin >> celltype >> xLow;
			//cerr << "spin: " << spin << " celltype: : " << celltype << 
			//     " xLow: " << xLow << endl;
			ASSERT_OR_THROW(string("PIF reader: xLow out of bounds : \n")+line, xLow >= 0 && xLow < dim.x);
			pif >> xHigh;
			ASSERT_OR_THROW(string("PIF reader: xHigh out of bounds : \n")+line, xHigh >= 0 && xHigh < dim.x);
			ASSERT_OR_THROW(string("PIF reader: xHigh is smaller than xLow : \n")+line, xHigh >= xLow); 
			pif >> yLow;
			ASSERT_OR_THROW(string("PIF reader: yLow out of bounds : \n")+line, yLow >= 0 && yLow < dim.y);
			pif >> yHigh;   
			ASSERT_OR_THROW(string("PIF reader: yHigh out of bounds : \n")+line, yHigh >= 0 && yHigh < dim.y);
			ASSERT_OR_THROW(string("PIF reader: yHigh is smaller than yLow : \n")+line, yHigh >= yLow);
			pif >> zLow;
			ASSERT_OR_THROW(string("PIF reader: zLow out of bounds : \n")+line, zLow >= 0 && zLow < dim.z);
			pif >> zHigh;
			ASSERT_OR_THROW(string("PIF reader: zHigh out of bounds: \n ")+line, zHigh >= 0 && zHigh < dim.z);
			ASSERT_OR_THROW(string("PIF reader: zHigh is smaller than xLow: \n")+line, zHigh >= zLow);

			if (spinMap.count(spin) != 0) // Spin multiply listed
			{
				for (cellPt.z = zLow; cellPt.z <= zHigh; cellPt.z++)
					for (cellPt.y = yLow; cellPt.y <= yHigh; cellPt.y++)
						for (cellPt.x = xLow; cellPt.x <= xHigh; cellPt.x++){
							cellFieldG->set(cellPt, cellFieldG->get(spinMap[spin]));
							potts->runSteppers(); //used to ensure that VolumeTracker Plugin step fcn gets called every time we do something to the fields
							//It is necessary to do it this way because steppers are called only when we are performing pixel copies
							// but if we initialize steppers are not called thus is you overwrite a cell here it will not get removed from
							//inventory unless you call steppers(VolumeTrackerPlugin) explicitely

						}
						//cerr << "2. Cell Type from cell->:  " <<(int)cell->type<< "\n";

			}
			else // First time for this spin, we need to create a new cell
			{
				spinMap[spin] = Point3D(xLow, yLow, zLow);
				cell = potts->createCellGSpecifiedIds(Point3D(xLow, yLow, zLow),spin);
				
				cell->type=potts->getAutomaton()->getTypeId(celltype);//first manually set cell type , then we reset it  via setType of transition Ptr  (transitionPtr is obsolete and not really used in most recent CC3D versions)
				potts->runSteppers(); //used to ensure that VolumeTracker Plugin step fcn gets called every time we do something to the fields
				//It is necessary to do it this way because steppers are called only when we are performing pixel copies
				// but if we initialize steppers are not called thus is you overwrite a cell here it will not get removed from
				//inventory unless you call steppers(VolumeTrackerPlugin) explicitely

				for (cellPt.z = zLow; cellPt.z <= zHigh; cellPt.z++)
					for (cellPt.y = yLow; cellPt.y <= yHigh; cellPt.y++)
						for (cellPt.x = xLow; cellPt.x <= xHigh; cellPt.x++){
							cellFieldG->set(cellPt, cell);
							potts->runSteppers(); //used to ensure that VolumeTracker Plugin step fcn gets called every time we do something to the fields
							//It is necessary to do it this way because steppers are called only when we are performing pixel copies
							// but if we initialize steppers are not called thus is you overwrite a cell here it will not get removed from
							//inventory unless you call steppers(VolumeTrackerPlugin) explicitely

						}


						typeTransitionPtr->setType( cell,potts->getAutomaton()->getTypeId(celltype));
						//cerr << "2. Cell Type from cell->:  " <<(int)cell->type<< "\n";

			}


		}
	}


}


