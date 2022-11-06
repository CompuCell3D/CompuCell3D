
#include <CompuCell3D/CC3D.h>

using namespace CompuCell3D;
using namespace std;

#include "PIFInitializer.h"
#include <Logger/CC3DLogger.h>

PIFInitializer::PIFInitializer() :
        potts(0), sim(0), pifname("") {}

PIFInitializer::PIFInitializer(string filename) :
        potts(0), sim(0), pifname(filename) {}

void PIFInitializer::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

    sim = simulator;

    bool pluginAlreadyRegisteredFlag;
    //this will load VolumeTracker plugin if it is not already loaded
    Plugin *plugin = Simulator::pluginManager.get("VolumeTracker",
                                                  &pluginAlreadyRegisteredFlag);
    if (!pluginAlreadyRegisteredFlag)
        plugin->init(simulator);

    pifname = _xmlData->getFirstElement("PIFName")->getText();

    std::string basePath = simulator->getBasePath();
    CC3D_Log(LOG_DEBUG) << "basePath=simulator->getBasePath()=" << simulator->getBasePath();
    if (basePath != "") {
        pifname = basePath + "/" + pifname;
    }


    potts = simulator->getPotts();
}

void PIFInitializer::start() {
	if (sim->getRestartEnabled()){
		return ;  // we will not initialize cells if restart flag is on
	}
	CC3D_Log(LOG_DEBUG) << "ppdPtr->pifname="<<pifname;

    std::ifstream piffile(pifname.c_str(), ios::in);
    CC3D_Log(LOG_DEBUG) << "opened pid file";
    if (!piffile.good())
        throw CC3DException(string("Could not open\n" + pifname + "\nMake sure it exists and is in correct directory"));
    WatchableField3D < CellG * > *cellFieldG = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();
    if (!cellFieldG) throw CC3DException("initField() Cell field cannot be null!");

    Dim3D dim = cellFieldG->getDim();
    CC3D_Log(LOG_DEBUG) << "THIS IS DIM FOR PIF "<<dim;

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
    CellG *cell;

	TypeTransition * typeTransitionPtr=potts->getTypeTransition();
	CC3D_Log(LOG_DEBUG) << "typeTransitionPtr="<<typeTransitionPtr;
    getline(piffile, line);
    istringstream pif(line);
    pif >> first >> second;
    CC3D_Log(LOG_DEBUG) << "First: " << first << " Second: " << second;
    if (second == "Clusters") {
        CC3D_Log(LOG_DEBUG) << "Clusters Included";
		while(getline(piffile,line)) {
			istringstream pif(line);
			pif >> clusterId>> spin >> celltype >> xLow;
					CC3D_Log(LOG_TRACE) << "  Cluster Id:  " <<clusterId<< "  Spin: " << spin
                           << "  Type: " <<  celltype;

            if (!(xLow >= 0 && xLow < dim.x)) throw CC3DException(string("PIF reader: xLow out of bounds : \n") + line);
            pif >> xHigh;
            if (!(xHigh >= 0 && xHigh < dim.x))
                throw CC3DException(string("PIF reader: xHigh out of bounds : \n") + line);
            if (xHigh < xLow) throw CC3DException(string("PIF reader: xHigh is smaller than xLow : \n") + line);
            pif >> yLow;
            if (!(yLow >= 0 && yLow < dim.y)) throw CC3DException(string("PIF reader: yLow out of bounds : \n") + line);
            pif >> yHigh;
            if (!(yHigh >= 0 && yHigh < dim.y))
                throw CC3DException(string("PIF reader: yHigh out of bounds : \n") + line);
            if (yHigh < yLow) throw CC3DException(string("PIF reader: yHigh is smaller than yLow : \n") + line);
            pif >> zLow;
            if (!(zLow >= 0 && zLow < dim.z)) throw CC3DException(string("PIF reader: zLow out of bounds : \n") + line);
            pif >> zHigh;
            if (!(zHigh >= 0 && zHigh < dim.z))
                throw CC3DException(string("PIF reader: zHigh out of bounds : \n") + line);
            if (zHigh < zLow) throw CC3DException(string("PIF reader: zHigh is smaller than xLow : \n") + line);
            if (spinMap.count(spin) != 0) // Spin multiply listed
            {
                for (cellPt.z = zLow; cellPt.z <= zHigh; cellPt.z++)
                    for (cellPt.y = yLow; cellPt.y <= yHigh; cellPt.y++)
                        for (cellPt.x = xLow; cellPt.x <= xHigh; cellPt.x++) {
                            cellFieldG->set(cellPt, cellFieldG->get(spinMap[spin]));
                            potts->runSteppers(); //used to ensure that VolumeTracker Plugin step fcn gets called every time we do something to the fields
                            //It is necessary to do it this way because steppers are called only when we are performing pixel copies
                            // but if we initialize steppers are not called thus is you overwrite a cell here it will not get removed from
                            //inventory unless you call steppers(VolumeTrackerPlugin) explicitly

                        }


            } else // First time for this spin, we need to create a new cell
            {
                spinMap[spin] = Point3D(xLow, yLow, zLow);
                cell = potts->createCellGSpecifiedIds(Point3D(xLow, yLow, zLow), spin, clusterId);

                cell->type = potts->getAutomaton()->getTypeId(
                        celltype);//first manually set cell type , then we reset it  via setType of transition Ptr  (transitionPtr is obsolete and not really used in most recent CC3D versions)

                potts->runSteppers(); //used to ensure that VolumeTracker Plugin step fcn gets called every time we do something to the fields
                //It is necessary to do it this way because steppers are called only when we are performing pixel copies
                // but if we initialize steppers are not called thus is you overwrite a cell here it will not get removed from
                //inventory unless you call steppers(VolumeTrackerPlugin) explicitly

                for (cellPt.z = zLow; cellPt.z <= zHigh; cellPt.z++)
                    for (cellPt.y = yLow; cellPt.y <= yHigh; cellPt.y++)
                        for (cellPt.x = xLow; cellPt.x <= xHigh; cellPt.x++) {
                            cellFieldG->set(cellPt, cell);
                            potts->runSteppers(); //used to ensure that VolumeTracker Plugin step fcn gets called every time we do something to the fields
                            //It is necessary to do it this way because steppers are called only when we are performing pixel copies
                            // but if we initialize steppers are not called thus is you overwrite a cell here it will not get removed from
                            //inventory unless you call steppers(VolumeTrackerPlugin) explicitly

                        }


                typeTransitionPtr->setType(cell, potts->getAutomaton()->getTypeId(celltype));


            }


        }
    } else {
        CC3D_Log(LOG_TRACE) << "Only Cell Types";
		pif >> xLow;
		int tmp = atoi(first.c_str());
		spin = tmp;
		celltype = second;
		CC3D_Log(LOG_TRACE) << "spin: " << spin << " celltype: : " << celltype <<
            " xLow: " << xLow;
        if (!(xLow >= 0 && xLow < dim.x)) throw CC3DException(string("PIF reader: xLow out of bounds : \n") + line);
        pif >> xHigh;
        if (!(xHigh >= 0 && xHigh < dim.x)) throw CC3DException(string("PIF reader: xHigh out of bounds : \n") + line);
        if (xHigh < xLow) throw CC3DException(string("PIF reader: xHigh is smaller than xLow : \n") + line);
        pif >> yLow;
        if (!(yLow >= 0 && yLow < dim.y)) throw CC3DException(string("PIF reader: yLow out of bounds : \n") + line);
        pif >> yHigh;
        if (!(yHigh >= 0 && yHigh < dim.y)) throw CC3DException(string("PIF reader: yHigh out of bounds : \n") + line);
        if (yHigh < yLow) throw CC3DException(string("PIF reader: yHigh is smaller than yLow : \n") + line);
        pif >> zLow;
        if (!(zLow >= 0 && zLow < dim.z)) throw CC3DException(string("PIF reader: zLow out of bounds : \n") + line);
        pif >> zHigh;
        if (!(zHigh >= 0 && zHigh < dim.z)) throw CC3DException(string("PIF reader: zHigh out of bounds : \n") + line);
        if (zHigh < zLow) throw CC3DException(string("PIF reader: zHigh is smaller than xLow : \n") + line);

        if (spinMap.count(spin) != 0) // Spin multiply listed
        {
            for (cellPt.z = zLow; cellPt.z <= zHigh; cellPt.z++)
                for (cellPt.y = yLow; cellPt.y <= yHigh; cellPt.y++)
                    for (cellPt.x = xLow; cellPt.x <= xHigh; cellPt.x++) {
                        cellFieldG->set(cellPt, cellFieldG->get(spinMap[spin]));
                        potts->runSteppers(); //used to ensure that VolumeTracker Plugin step fcn gets called every time we do something to the fields
                        //It is necessary to do it this way because steppers are called only when we are performing pixel copies
                        // but if we initialize steppers are not called thus is you overwrite a cell here it will not get removed from
                        //inventory unless you call steppers(VolumeTrackerPlugin) explicitely

                    }
        } else // First time for this spin, we need to create a new cell
        {
            spinMap[spin] = Point3D(xLow, yLow, zLow);
            cell = potts->createCellGSpecifiedIds(Point3D(xLow, yLow, zLow), spin);
            cell->type = potts->getAutomaton()->getTypeId(
                    celltype);//first manually set cell type , then we reset it  via setType of transition Ptr  (transitionPtr is obsolete and not really used in most recent CC3D versions)

            potts->runSteppers(); //used to ensure that VolumeTracker Plugin step fcn gets called every time we do something to the fields
            //It is necessary to do it this way because steppers are called only when we are performing pixel copies
            // but if we initialize steppers are not called thus is you overwrite a cell here it will not get removed from
            //inventory unless you call steppers(VolumeTrackerPlugin) explicitly

            for (cellPt.z = zLow; cellPt.z <= zHigh; cellPt.z++)
                for (cellPt.y = yLow; cellPt.y <= yHigh; cellPt.y++)
                    for (cellPt.x = xLow; cellPt.x <= xHigh; cellPt.x++) {
                        cellFieldG->set(cellPt, cell);
                        potts->runSteppers(); //used to ensure that VolumeTracker Plugin step fcn gets called every time we do something to the fields
                        //It is necessary to do it this way because steppers are called only when we are performing pixel copies
                        // but if we initialize steppers are not called thus is you overwrite a cell here it will not get removed from
                        //inventory unless you call steppers(VolumeTrackerPlugin) explicitly

                    }


            typeTransitionPtr->setType(cell, potts->getAutomaton()->getTypeId(celltype));
        }
        while (getline(piffile, line)) {

            istringstream pif(line);
            pif >> spin >> celltype >> xLow;
            if (!(xLow >= 0 && xLow < dim.x)) throw CC3DException(string("PIF reader: xLow out of bounds : \n") + line);
            pif >> xHigh;
            if (!(xHigh >= 0 && xHigh < dim.x))
                throw CC3DException(string("PIF reader: xHigh out of bounds : \n") + line);
            if (xHigh < xLow) throw CC3DException(string("PIF reader: xHigh is smaller than xLow : \n") + line);
            pif >> yLow;
            if (!(yLow >= 0 && yLow < dim.y)) throw CC3DException(string("PIF reader: yLow out of bounds : \n") + line);
            pif >> yHigh;
            if (!(yHigh >= 0 && yHigh < dim.y))
                throw CC3DException(string("PIF reader: yHigh out of bounds : \n") + line);
            if (yHigh < yLow) throw CC3DException(string("PIF reader: yHigh is smaller than yLow : \n") + line);
            pif >> zLow;
            if (!(zLow >= 0 && zLow < dim.z)) throw CC3DException(string("PIF reader: zLow out of bounds : \n") + line);
            pif >> zHigh;
            if (!(zHigh >= 0 && zHigh < dim.z))
                throw CC3DException(string("PIF reader: zHigh out of bounds: \n ") + line);
            if (zHigh < zLow) throw CC3DException(string("PIF reader: zHigh is smaller than xLow: \n") + line);

            if (spinMap.count(spin) != 0) // Spin multiply listed
            {
                for (cellPt.z = zLow; cellPt.z <= zHigh; cellPt.z++)
                    for (cellPt.y = yLow; cellPt.y <= yHigh; cellPt.y++)
                        for (cellPt.x = xLow; cellPt.x <= xHigh; cellPt.x++) {
                            cellFieldG->set(cellPt, cellFieldG->get(spinMap[spin]));
                            potts->runSteppers(); //used to ensure that VolumeTracker Plugin step fcn gets called every time we do something to the fields
                            //It is necessary to do it this way because steppers are called only when we are performing pixel copies
                            // but if we initialize steppers are not called thus is you overwrite a cell here it will not get removed from
                            //inventory unless you call steppers(VolumeTrackerPlugin) explicitly

                        }


            } else // First time for this spin, we need to create a new cell
            {
                spinMap[spin] = Point3D(xLow, yLow, zLow);
                cell = potts->createCellGSpecifiedIds(Point3D(xLow, yLow, zLow), spin);

                cell->type = potts->getAutomaton()->getTypeId(
                        celltype);//first manually set cell type , then we reset it  via setType of transition Ptr  (transitionPtr is obsolete and not really used in most recent CC3D versions)
                potts->runSteppers(); //used to ensure that VolumeTracker Plugin step fcn gets called every time we do something to the fields
                //It is necessary to do it this way because steppers are called only when we are performing pixel copies
                // but if we initialize steppers are not called thus is you overwrite a cell here it will not get removed from
                //inventory unless you call steppers(VolumeTrackerPlugin) explicitly

                for (cellPt.z = zLow; cellPt.z <= zHigh; cellPt.z++)
                    for (cellPt.y = yLow; cellPt.y <= yHigh; cellPt.y++)
                        for (cellPt.x = xLow; cellPt.x <= xHigh; cellPt.x++) {
                            cellFieldG->set(cellPt, cell);
                            potts->runSteppers(); //used to ensure that VolumeTracker Plugin step fcn gets called every time we do something to the fields
                            //It is necessary to do it this way because steppers are called only when we are performing pixel copies
                            // but if we initialize steppers are not called thus is you overwrite a cell here it will not get removed from
                            //inventory unless you call steppers(VolumeTrackerPlugin) explicitly

                        }


                typeTransitionPtr->setType(cell, potts->getAutomaton()->getTypeId(celltype));

            }


        }
    }


}


