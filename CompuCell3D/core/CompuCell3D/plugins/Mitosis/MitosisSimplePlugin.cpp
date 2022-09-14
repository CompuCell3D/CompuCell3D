
#include <CompuCell3D/CC3D.h>

using namespace CompuCell3D;
using namespace std;


#include "MitosisSimplePlugin.h"
#include "CompuCell3D/plugins/PixelTracker/PixelTrackerPlugin.h"
#include "CompuCell3D/plugins/PixelTracker/PixelTracker.h"

MitosisSimplePlugin::MitosisSimplePlugin() :
        MitosisPlugin() {
    potts = 0;
    doDirectionalMitosis2DPtr = 0;
    getOrientationVectorsMitosis2DPtr = 0;

}

MitosisSimplePlugin::~MitosisSimplePlugin() {}

void MitosisSimplePlugin::setDivideAlongMinorAxis() {
    divideAlongMinorAxisFlag = true;
    divideAlongMajorAxisFlag = false;

}

void MitosisSimplePlugin::setDivideAlongMajorAxis() {
    divideAlongMinorAxisFlag = false;
    divideAlongMajorAxisFlag = true;
}

void MitosisSimplePlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

    potts = simulator->getPotts();
    bool pluginAlreadyRegisteredFlag;
    Plugin *plugin = Simulator::pluginManager.get("VolumeTracker",
                                                  &pluginAlreadyRegisteredFlag); //this will load VolumeTracker plugin if it is not already loaded
    CC3D_Log(LOG_DEBUG) << "GOT HERE BEFORE CALLING INIT";;
    if (!pluginAlreadyRegisteredFlag)
        plugin->init(simulator);

    Plugin *pluginCOM = Simulator::pluginManager.get("CenterOfMass",
                                                     &pluginAlreadyRegisteredFlag); //this will load CenterOfMass plugin if it is not already loaded
    CC3D_Log(LOG_DEBUG) << "GOT HERE BEFORE CALLING INIT";
    if (!pluginAlreadyRegisteredFlag)
        pluginCOM->init(simulator);


    pixelTrackerPlugin = (PixelTrackerPlugin *) Simulator::pluginManager.get("PixelTracker",
                                                                             &pluginAlreadyRegisteredFlag); //this will load VolumeTracker plugin if it is not already loaded
    if (!pluginAlreadyRegisteredFlag)
        pixelTrackerPlugin->init(simulator);

    pixelTrackerAccessorPtr = pixelTrackerPlugin->getPixelTrackerAccessorPtr();

    Dim3D fieldDim = simulator->getPotts()->getCellFieldG()->getDim();

    if (fieldDim.x != 1 && fieldDim.y != 1 && fieldDim.z != 1) {
        flag3D = true;
    } else {
        if (fieldDim.x == 1) {
            doDirectionalMitosis2DPtr = &MitosisSimplePlugin::doDirectionalMitosis2D_yz;
            getOrientationVectorsMitosis2DPtr = &MitosisSimplePlugin::getOrientationVectorsMitosis2D_yz;
            flag3D = false;
        } else if (fieldDim.y == 1) {
            doDirectionalMitosis2DPtr = &MitosisSimplePlugin::doDirectionalMitosis2D_xz;
            getOrientationVectorsMitosis2DPtr = &MitosisSimplePlugin::getOrientationVectorsMitosis2D_xz;
            flag3D = false;
        } else if (fieldDim.z == 1) {
            doDirectionalMitosis2DPtr = &MitosisSimplePlugin::doDirectionalMitosis2D_xy;
            getOrientationVectorsMitosis2DPtr = &MitosisSimplePlugin::getOrientationVectorsMitosis2D_xy;
            flag3D = false;
        }
    }

    pUtils = simulator->getParallelUtils();
    unsigned int maxNumberOfWorkNodes = pUtils->getMaxNumberOfWorkNodesPotts();
    childCellVec.assign(maxNumberOfWorkNodes, 0);
    parentCellVec.assign(maxNumberOfWorkNodes, 0);
    splitPtVec.assign(maxNumberOfWorkNodes, Point3D());
    splitVec.assign(maxNumberOfWorkNodes, false);
    onVec.assign(maxNumberOfWorkNodes, false);
    mitosisFlagVec.assign(maxNumberOfWorkNodes, false);

   turnOn(); //this can be called only after vectors have been allocated
	CC3D_Log(LOG_DEBUG) << "maxNumberOfWorkNodes="<<maxNumberOfWorkNodes;


    boundaryStrategy = BoundaryStrategy::getInstance();
    maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(5);


}

void MitosisSimplePlugin::handleEvent(CC3DEvent &_event) {
    if (_event.id == CHANGE_NUMBER_OF_WORK_NODES) {
        unsigned int maxNumberOfWorkNodes = pUtils->getMaxNumberOfWorkNodesPotts();
        childCellVec.assign(maxNumberOfWorkNodes, 0);
        parentCellVec.assign(maxNumberOfWorkNodes, 0);
        splitPtVec.assign(maxNumberOfWorkNodes, Point3D());
        splitVec.assign(maxNumberOfWorkNodes, false);
        onVec.assign(maxNumberOfWorkNodes, false);
        mitosisFlagVec.assign(maxNumberOfWorkNodes, false);

        turnOn(); //this can be called only after vectors have been allocated

    }

}


void MitosisSimplePlugin::setMitosisFlag(bool _flag) {
    mitosisFlagVec[pUtils->getCurrentWorkNodeNumber()] = _flag;
}

bool MitosisSimplePlugin::getMitosisFlag() {
    return mitosisFlagVec[pUtils->getCurrentWorkNodeNumber()];
}


bool MitosisSimplePlugin::doDirectionalMitosis() {


    if (flag3D) {

        return doDirectionalMitosis3D();
    } else {

        return (this->*doDirectionalMitosis2DPtr)();
    }
}

OrientationVectorsMitosis MitosisSimplePlugin::getOrientationVectorsMitosis(CellG *_cell) {
    if (flag3D) {

        return getOrientationVectorsMitosis3D(_cell);
    } else {
        return (this->*getOrientationVectorsMitosis2DPtr)(_cell);
    }
}

OrientationVectorsMitosis MitosisSimplePlugin::getOrientationVectorsMitosis2D_xy(CellG *cell) {
    double xcm = cell->xCM / (float) cell->volume;
    double ycm = cell->yCM / (float) cell->volume;
    double zcm = cell->zCM / (float) cell->volume;

    //first calculate and diagonalize inertia tensor
    vector <vector<double>> inertiaTensor(2, vector<double>(2, 0.0));


    set <PixelTrackerData> cellPixels = pixelTrackerAccessorPtr->get(cell->extraAttribPtr)->pixelSet;
    for (set<PixelTrackerData>::iterator sitr = cellPixels.begin(); sitr != cellPixels.end(); ++sitr) {
        Coordinates3D<double> pixelTrans= boundaryStrategy->calculatePointCoordinates(sitr->pixel);
			inertiaTensor[0][0]+=(pixelTrans.y-ycm)*(pixelTrans.y-ycm);
			inertiaTensor[0][1]+=-(pixelTrans.x-xcm)*(pixelTrans.y-ycm);
			inertiaTensor[1][1]+=(pixelTrans.x-xcm)*(pixelTrans.x-xcm);
		}
		inertiaTensor[1][0]=inertiaTensor[0][1];

    double radical = 0.5 *
                     sqrt((inertiaTensor[0][0] - inertiaTensor[1][1]) * (inertiaTensor[0][0] - inertiaTensor[1][1]) +
                          4.0 * inertiaTensor[0][1] * inertiaTensor[0][1]);
    double lMin = 0.5 * (inertiaTensor[0][0] + inertiaTensor[1][1]) - radical;
    double lMax = 0.5 * (inertiaTensor[0][0] + inertiaTensor[1][1]) + radical;

    //orientationVec points along semiminor axis (it corresponds to larger eigenvalue)
    Coordinates3D<double> orientationVec;
    if (inertiaTensor[0][1] != 0.0) {

        orientationVec = Coordinates3D<double>(inertiaTensor[0][1], lMax - inertiaTensor[0][0], 0.0);
        double length = sqrt(orientationVec.x * orientationVec.x + orientationVec.y * orientationVec.y +
                             orientationVec.z * orientationVec.z);
        orientationVec.x /= length;
        orientationVec.y /= length;
        orientationVec.z /= length;
    } else {
        if (inertiaTensor[0][0] > inertiaTensor[1][1])
            orientationVec = Coordinates3D<double>(0.0, 1.0, 0.0);
        else
            orientationVec = Coordinates3D<double>(1.0, 0.0, 0.0);
    }


    OrientationVectorsMitosis orientationVectorsMitosis;
    orientationVectorsMitosis.semiminorVec = orientationVec;
    orientationVectorsMitosis.semimajorVec.x = -orientationVectorsMitosis.semiminorVec.y;
    orientationVectorsMitosis.semimajorVec.y = orientationVectorsMitosis.semiminorVec.x;

    return orientationVectorsMitosis;
}

bool MitosisSimplePlugin::doDirectionalMitosis2D_xy() {

    int currentWorkNodeNumber = pUtils->getCurrentWorkNodeNumber();

    // simplifying access to vectorized class variables
    short &split = splitVec[currentWorkNodeNumber];
    short &on = onVec[currentWorkNodeNumber];
    CellG *&childCell = childCellVec[currentWorkNodeNumber];
    CellG *&parentCell = parentCellVec[currentWorkNodeNumber];
    Point3D &splitPt = splitPtVec[currentWorkNodeNumber];

    //this implementation is valid in 2D only
    if (split && on) {
        split = false;

        WatchableField3D < CellG * > *cellField = (WatchableField3D < CellG * > *)
        potts->getCellFieldG();
        //reseting poiters to parent and child cell - neessary otherwise may get some strange side effects when mitisis is aborted
        childCell = 0;
        parentCell = 0;

        CellG *cell = cellField->get(splitPt);//cells that is being divided
        parentCell = cell;

        double xcm = cell->xCM / (float) cell->volume;
        double ycm = cell->yCM / (float) cell->volume;
        double zcm = cell->zCM / (float) cell->volume;

        set <PixelTrackerData> cellPixels = pixelTrackerAccessorPtr->get(cell->extraAttribPtr)->pixelSet;

        OrientationVectorsMitosis orientationVectorsMitosis = getOrientationVectorsMitosis2D_xy(cell);
        Coordinates3D<double> orientationVec = orientationVectorsMitosis.semiminorVec;

		// assume the following form of the equation of the straight line passing through COM (xcm,ycm,zcm) of cell being divided
		//y=a*x+b;
		double a;
		double b;
		//determining coefficients of the straight line passing through
		if(divideAlongMinorAxisFlag){


            a = orientationVec.y / orientationVec.x;
            b = ycm - xcm * a;

            if (a != a) {//a is Nan - will happen when orientationVec.x is 0.0 thus minor axis is along y axis
                a = 0.0;
                b = 0.0;
            }
        }

        if (divideAlongMajorAxisFlag) {
            if(orientationVec.y==0.0){//then perpendicular vector (major axis) is along y axis meaning:
				a=0.0;
				b=0.0;

            } else {
                a = -orientationVec.x / orientationVec.y;
                b = ycm - xcm * a;
            }
        }

        //now do the division

        if (a == 0.0 && b == 0.0) {//division along y axis

            for (set<PixelTrackerData>::iterator sitr = cellPixels.begin(); sitr != cellPixels.end(); ++sitr) {
                Coordinates3D<double> pixelTrans = boundaryStrategy->calculatePointCoordinates(sitr->pixel);
                if (pixelTrans.x < xcm) {
                    if (!childCell) {
                        childCell = potts->createCellG(sitr->pixel);
                    } else {
                        cellField->set(sitr->pixel, childCell);
                    }
                }
            }

        } else {//division will be done along axis different than y axis
            int parentCellVolume=0;
			for(set<PixelTrackerData>::iterator sitr=cellPixels.begin() ; sitr != cellPixels.end() ;++sitr){
				Coordinates3D<double> pixelTrans= boundaryStrategy->calculatePointCoordinates(sitr->pixel);
				if(pixelTrans.y <= a*pixelTrans.x+b){

					if(!childCell){
						childCell = potts->createCellG(sitr->pixel);
					}else{
                        cellField->set(sitr->pixel, childCell);
                    }
                } else {

                    parentCellVolume++;
                }

			}
		}

        //if childCell was created this means mitosis was sucessful. If child cell was not created there was no mitosis
        if (childCell)
            return true;
        else
            return false;
    }
}


OrientationVectorsMitosis MitosisSimplePlugin::getOrientationVectorsMitosis2D_xz(CellG *cell) {
    double xcm = cell->xCM / (float) cell->volume;
    double ycm = cell->yCM / (float) cell->volume;
    double zcm = cell->zCM / (float) cell->volume;

    //first calculate and diagonalize inertia tensor
    vector <vector<double>> inertiaTensor(2, vector<double>(2, 0.0));

    set <PixelTrackerData> cellPixels = pixelTrackerAccessorPtr->get(cell->extraAttribPtr)->pixelSet;
    for (set<PixelTrackerData>::iterator sitr = cellPixels.begin(); sitr != cellPixels.end(); ++sitr) {
        Coordinates3D<double> pixelTrans= boundaryStrategy->calculatePointCoordinates(sitr->pixel);
			inertiaTensor[0][0]+=(pixelTrans.z-zcm)*(pixelTrans.z-zcm);
			inertiaTensor[0][1]+=-(pixelTrans.x-xcm)*(pixelTrans.z-zcm);
			inertiaTensor[1][1]+=(pixelTrans.x-xcm)*(pixelTrans.x-xcm);
		}
		inertiaTensor[1][0]=inertiaTensor[0][1];

    double radical = 0.5 *
                     sqrt((inertiaTensor[0][0] - inertiaTensor[1][1]) * (inertiaTensor[0][0] - inertiaTensor[1][1]) +
                          4.0 * inertiaTensor[0][1] * inertiaTensor[0][1]);
    double lMin = 0.5 * (inertiaTensor[0][0] + inertiaTensor[1][1]) - radical;
    double lMax = 0.5 * (inertiaTensor[0][0] + inertiaTensor[1][1]) + radical;

    //orientationVec points along semiminor axis (it corresponds to larger eigenvalue)
    Coordinates3D<double> orientationVec;
    if (inertiaTensor[0][1] != 0.0) {

        orientationVec = Coordinates3D<double>(inertiaTensor[0][1], 0.0, lMax - inertiaTensor[0][0]);
        double length = sqrt(orientationVec.x * orientationVec.x + orientationVec.y * orientationVec.y +
                             orientationVec.z * orientationVec.z);
        orientationVec.x /= length;
        orientationVec.y /= length;
        orientationVec.z /= length;
    } else {
        if (inertiaTensor[0][0] > inertiaTensor[1][1])
            orientationVec = Coordinates3D<double>(0.0, 0.0, 1.0);
        else
            orientationVec = Coordinates3D<double>(1.0, 0.0, 0.0);
    }

    OrientationVectorsMitosis orientationVectorsMitosis;
    orientationVectorsMitosis.semiminorVec = orientationVec;
    orientationVectorsMitosis.semimajorVec.x = -orientationVectorsMitosis.semiminorVec.z;
    orientationVectorsMitosis.semimajorVec.z = orientationVectorsMitosis.semiminorVec.x;

    return orientationVectorsMitosis;


}

bool MitosisSimplePlugin::doDirectionalMitosis2D_xz() {
    int currentWorkNodeNumber = pUtils->getCurrentWorkNodeNumber();

    // simplifying access to vectorized class variables
    short &split = splitVec[currentWorkNodeNumber];
    short &on = onVec[currentWorkNodeNumber];
    CellG *&childCell = childCellVec[currentWorkNodeNumber];
    CellG *&parentCell = parentCellVec[currentWorkNodeNumber];
    Point3D &splitPt = splitPtVec[currentWorkNodeNumber];
    //this implementation is valid in 2D only
    if (split && on) {
        split = false;

        WatchableField3D < CellG * > *cellField = (WatchableField3D < CellG * > *)
        potts->getCellFieldG();
        //reseting poiters to parent and child cell - neessary otherwise may get some strange side effects when mitisis is aborted
        childCell = 0;
        parentCell = 0;

        CellG *cell = cellField->get(splitPt);//cells that is being divided
        parentCell = cell;
        double xcm=cell->xCM/(float)cell->volume;
		double ycm=cell->yCM/(float)cell->volume;
		double zcm=cell->zCM/(float)cell->volume;
		
		set<PixelTrackerData> cellPixels=pixelTrackerAccessorPtr->get(cell->extraAttribPtr)->pixelSet;

        OrientationVectorsMitosis orientationVectorsMitosis = getOrientationVectorsMitosis2D_xz(cell);
        Coordinates3D<double> orientationVec = orientationVectorsMitosis.semiminorVec;

		//once we know orientation vector corresponding to bigger eigenvalue (pointing along semiminor axis) we may divide cell
		//along major or minor axis

		// assume the following form of the equation of the straight line passing through COM (xcm,ycm,zcm) of cell being divided
		//z=a*x+b;
		double a;
		double b;
		//determining coefficients of the straight line passing through
		if(divideAlongMinorAxisFlag){


            a = orientationVec.z / orientationVec.x;
            b = zcm - xcm * a;

            if (a != a) {//a is Nan - will happen when orientationVec.x is 0.0 thus minor axis is along y axis
                a = 0.0;
                b = 0.0;
            }
        }

        if (divideAlongMajorAxisFlag) {
            if(orientationVec.z==0.0){//then perpendicular vector (major axis) is along z axis meaning:
				a=0.0;
				b=0.0;

			} else{
				a=-orientationVec.x/orientationVec.z;
				b=zcm-xcm*a;
			}
		}
		//now do the division

        if (a == 0.0 && b == 0.0) {//division along y axis

            for (set<PixelTrackerData>::iterator sitr = cellPixels.begin(); sitr != cellPixels.end(); ++sitr) {
                Coordinates3D<double> pixelTrans = boundaryStrategy->calculatePointCoordinates(sitr->pixel);
                if (pixelTrans.x < xcm) {
                    if (!childCell) {
                        childCell = potts->createCellG(sitr->pixel);
                    } else {
                        cellField->set(sitr->pixel, childCell);
                    }
                }
            }

        } else {//division will be done along axis different than y axis
            int parentCellVolume=0;
			for(set<PixelTrackerData>::iterator sitr=cellPixels.begin() ; sitr != cellPixels.end() ;++sitr){
				Coordinates3D<double> pixelTrans= boundaryStrategy->calculatePointCoordinates(sitr->pixel);
				if(pixelTrans.z <= a*pixelTrans.x+b){

					if(!childCell){
						childCell = potts->createCellG(sitr->pixel);
					}else{
                        cellField->set(sitr->pixel, childCell);
                    }
                } else {

                    parentCellVolume++;
                }

			}
		}

        //if childCell was created this means mitosis was sucessful. If child cell was not created there was no mitosis
        if (childCell)
            return true;
        else
            return false;
    }
}

OrientationVectorsMitosis MitosisSimplePlugin::getOrientationVectorsMitosis2D_yz(CellG *cell) {
    double xcm = cell->xCM / (float) cell->volume;
    double ycm = cell->yCM / (float) cell->volume;
    double zcm = cell->zCM / (float) cell->volume;

    //first calculate and diagonalize inertia tensor
    vector <vector<double>> inertiaTensor(2, vector<double>(2, 0.0));

    set <PixelTrackerData> cellPixels = pixelTrackerAccessorPtr->get(cell->extraAttribPtr)->pixelSet;
    for (set<PixelTrackerData>::iterator sitr = cellPixels.begin(); sitr != cellPixels.end(); ++sitr) {
        Coordinates3D<double> pixelTrans = boundaryStrategy->calculatePointCoordinates(sitr->pixel);
        inertiaTensor[0][0]+=(pixelTrans.z-zcm)*(pixelTrans.z-zcm);
			inertiaTensor[0][1]+=-(pixelTrans.y-ycm)*(pixelTrans.z-zcm);
			inertiaTensor[1][1]+=(pixelTrans.y-ycm)*(pixelTrans.y-ycm);
		}
		inertiaTensor[1][0]=inertiaTensor[0][1];

    double radical = 0.5 *
                     sqrt((inertiaTensor[0][0] - inertiaTensor[1][1]) * (inertiaTensor[0][0] - inertiaTensor[1][1]) +
                          4.0 * inertiaTensor[0][1] * inertiaTensor[0][1]);
    double lMin = 0.5 * (inertiaTensor[0][0] + inertiaTensor[1][1]) - radical;
    double lMax = 0.5 * (inertiaTensor[0][0] + inertiaTensor[1][1]) + radical;

    //orientationVec points along semiminor axis (it corresponds to larger eigenvalue)
    Coordinates3D<double> orientationVec;
    if (inertiaTensor[0][1] != 0.0) {

        orientationVec = Coordinates3D<double>(0.0, inertiaTensor[0][1], lMax - inertiaTensor[0][0]);
        double length = sqrt(orientationVec.x * orientationVec.x + orientationVec.y * orientationVec.y +
                             orientationVec.z * orientationVec.z);
        orientationVec.x /= length;
        orientationVec.y /= length;
        orientationVec.z /= length;
    } else {
        if (inertiaTensor[0][0] > inertiaTensor[1][1])
            orientationVec = Coordinates3D<double>(0.0, 0.0, 1.0);
        else
            orientationVec = Coordinates3D<double>(0.0, 1.0, 0.0);
    }

    OrientationVectorsMitosis orientationVectorsMitosis;
    orientationVectorsMitosis.semiminorVec = orientationVec;
    orientationVectorsMitosis.semimajorVec.y = -orientationVectorsMitosis.semiminorVec.z;
    orientationVectorsMitosis.semimajorVec.z = orientationVectorsMitosis.semiminorVec.y;

    return orientationVectorsMitosis;


}

bool MitosisSimplePlugin::doDirectionalMitosis2D_yz() {
    int currentWorkNodeNumber = pUtils->getCurrentWorkNodeNumber();

    // simplifying access to vectorized class variables
    short &split = splitVec[currentWorkNodeNumber];
    short &on = onVec[currentWorkNodeNumber];
    CellG *&childCell = childCellVec[currentWorkNodeNumber];
    CellG *&parentCell = parentCellVec[currentWorkNodeNumber];
    Point3D &splitPt = splitPtVec[currentWorkNodeNumber];

    //this implementation is valid in 2D only
    if (split && on) {
        split = false;

        WatchableField3D < CellG * > *cellField = (WatchableField3D < CellG * > *)
        potts->getCellFieldG();
        //reseting poiters to parent and child cell - neessary otherwise may get some strange side effects when mitisis is aborted
        childCell = 0;
        parentCell = 0;

        CellG *cell = cellField->get(splitPt);//cells that is being divided
        parentCell = cell;

        double xcm = cell->xCM / (float) cell->volume;
        double ycm = cell->yCM / (float) cell->volume;
        double zcm = cell->zCM / (float) cell->volume;

        set <PixelTrackerData> cellPixels = pixelTrackerAccessorPtr->get(cell->extraAttribPtr)->pixelSet;

        OrientationVectorsMitosis orientationVectorsMitosis = getOrientationVectorsMitosis2D_xz(cell);
        Coordinates3D<double> orientationVec = orientationVectorsMitosis.semiminorVec;

		//once we know orientation vector corresponding to bigger eigenvalue (pointing along semiminor axis) we may divide cell
		//along major or minor axis

		// assume the following form of the equation of the straight line passing through COM (xcm,ycm,zcm) of cell being divided
		//z=a*y+b;
		double a;
		double b;
		//determining coefficients of the straight line passing through
		if(divideAlongMinorAxisFlag){


            a = orientationVec.z / orientationVec.y;
            b = zcm - ycm * a;

            if (a != a) {//a is Nan - will happen when orientationVec.x is 0.0 thus minor axis is along y axis
                a = 0.0;
                b = 0.0;
            }
        }

        if (divideAlongMajorAxisFlag) {
            if(orientationVec.z==0.0){//then perpendicular vector (major axis) is along z axis meaning:
				a=0.0;
				b=0.0;

			} else{
				a=-orientationVec.y/orientationVec.z;
				b=zcm-ycm*a;
			}
		}
		//now do the division

        if (a == 0.0 && b == 0.0) {//division along y axis

            for (set<PixelTrackerData>::iterator sitr = cellPixels.begin(); sitr != cellPixels.end(); ++sitr) {
                Coordinates3D<double> pixelTrans = boundaryStrategy->calculatePointCoordinates(sitr->pixel);
                if (pixelTrans.y < ycm) {
                    if (!childCell) {
                        childCell = potts->createCellG(sitr->pixel);
                    } else {
                        cellField->set(sitr->pixel, childCell);
                    }
                }
            }

        } else {//division will be done along axis different than y axis
            int parentCellVolume=0;
			for(set<PixelTrackerData>::iterator sitr=cellPixels.begin() ; sitr != cellPixels.end() ;++sitr){
				Coordinates3D<double> pixelTrans= boundaryStrategy->calculatePointCoordinates(sitr->pixel);
				if(pixelTrans.z <= a*pixelTrans.y+b){

					if(!childCell){
						childCell = potts->createCellG(sitr->pixel);
					}else{
                        cellField->set(sitr->pixel, childCell);
                    }
                } else {

                    parentCellVolume++;
                }

			}
		}

        //if childCell was created this means mitosis was sucessful. If child cell was not created there was no mitosis
        if (childCell)
            return true;
        else
            return false;
    }
}

OrientationVectorsMitosis MitosisSimplePlugin::getOrientationVectorsMitosis3D(CellG *cell) {

    double xcm = cell->xCM / (float) cell->volume;
    double ycm = cell->yCM / (float) cell->volume;
    double zcm = cell->zCM / (float) cell->volume;

    //first calculate and diagonalize inertia tensor
    vector <vector<double>> inertiaTensor(3, vector<double>(3, 0.0));

    set <PixelTrackerData> cellPixels = pixelTrackerAccessorPtr->get(cell->extraAttribPtr)->pixelSet;
    for (set<PixelTrackerData>::iterator sitr = cellPixels.begin(); sitr != cellPixels.end(); ++sitr) {
        Coordinates3D<double> pixelTrans = boundaryStrategy->calculatePointCoordinates(sitr->pixel);
        inertiaTensor[0][0]+=(pixelTrans.y-ycm)*(pixelTrans.y-ycm)+(pixelTrans.z-zcm)*(pixelTrans.z-zcm);
			inertiaTensor[0][1]+=-(pixelTrans.x-xcm)*(pixelTrans.y-ycm);
			inertiaTensor[0][2]+=-(pixelTrans.x-xcm)*(pixelTrans.z-zcm);
			inertiaTensor[1][1]+=(pixelTrans.x-xcm)*(pixelTrans.x-xcm)+(pixelTrans.z-zcm)*(pixelTrans.z-zcm);
			inertiaTensor[1][2]+=-(pixelTrans.y-ycm)*(pixelTrans.z-zcm);
			inertiaTensor[2][2]+=(pixelTrans.x-xcm)*(pixelTrans.x-xcm)+(pixelTrans.y-ycm)*(pixelTrans.y-ycm);
		}
		inertiaTensor[1][0]=inertiaTensor[0][1];
		inertiaTensor[2][0]=inertiaTensor[0][2];
		inertiaTensor[2][1]=inertiaTensor[1][2];
		
	 //Finding eigenvalues
	 vector<double> aCoeff(4,0.0);
	 vector<complex<double> > roots;
	 
	 //initialize coefficients of cubic eq used to find eigenvalues of inertia tensor - before pixel copy
	 aCoeff[0]=-1.0;

    aCoeff[1] = inertiaTensor[0][0] + inertiaTensor[1][1] + inertiaTensor[2][2];

    aCoeff[2] = inertiaTensor[0][1] * inertiaTensor[0][1] + inertiaTensor[0][2] * inertiaTensor[0][2] +
                inertiaTensor[1][2] * inertiaTensor[1][2]
                - inertiaTensor[0][0] * inertiaTensor[1][1] - inertiaTensor[0][0] * inertiaTensor[2][2] -
                inertiaTensor[1][1] * inertiaTensor[2][2];

    aCoeff[3] = inertiaTensor[0][0] * inertiaTensor[1][1] * inertiaTensor[2][2] +
                2 * inertiaTensor[0][1] * inertiaTensor[0][2] * inertiaTensor[1][2]
                - inertiaTensor[0][0] * inertiaTensor[1][2] * inertiaTensor[1][2]
                - inertiaTensor[1][1] * inertiaTensor[0][2] * inertiaTensor[0][2]
                - inertiaTensor[2][2] * inertiaTensor[0][1] * inertiaTensor[0][1];

    roots = solveCubicEquationRealCoeeficients(aCoeff);


    vector <Coordinates3D<double>> eigenvectors(3);

	 for (int i = 0 ; i < 3 ; ++i){
		 eigenvectors[i].x=(inertiaTensor[0][2]*(inertiaTensor[1][1]-roots[i].real())-inertiaTensor[1][2]*inertiaTensor[0][1])/
		 (inertiaTensor[1][2]*(inertiaTensor[0][0]-roots[i].real())-inertiaTensor[0][1]*inertiaTensor[0][2]);
		 eigenvectors[i].y=1.0;
		 eigenvectors[i].z = (inertiaTensor[0][2]*eigenvectors[i].x+inertiaTensor[1][2]*eigenvectors[i].y)/
		 (roots[i].real()-inertiaTensor[2][2]) ;
		 if(eigenvectors[i].x!=eigenvectors[i].x || eigenvectors[i].y!=eigenvectors[i].y || eigenvectors[i].z!=eigenvectors[i].z){
			OrientationVectorsMitosis orientationVectorsMitosis;
			return orientationVectorsMitosis;//simply dont do mitosis if any of the eigenvector component is NaN
		 }
	 }



    //finding semiaxes of the ellipsoid
    //Ixx=m/5.0*(a_y^2+a_z^2) - andy cyclical permutations for other coordinate combinations
    //a_x,a_y,a_z are lengths of semiaxes of the allipsoid
    // We can invert above system of equations to get:
    vector<double> axes(3, 0.0);

    axes[0] = sqrt((2.5 / cell->volume) *
                   (roots[1].real() + roots[2].real() - roots[0].real()));//corresponds to first eigenvalue
    axes[1] = sqrt((2.5 / cell->volume) *
                   (roots[0].real() + roots[2].real() - roots[1].real()));//corresponds to second eigenvalue
    axes[2] = sqrt((2.5 / cell->volume) *
                   (roots[0].real() + roots[1].real() - roots[2].real()));//corresponds to third eigenvalue

    vector <pair<double, int>> sortedAxes(3);
    sortedAxes[0] = make_pair(axes[0], 0);
    sortedAxes[1] = make_pair(axes[1], 1);
    sortedAxes[2] = make_pair(axes[2], 2);

    //sorting semiaxes according the their lengths (shortest first)
    //sort(axes.begin(),axes.end());
    sort(sortedAxes.begin(),
         sortedAxes.end()); //by keeping track of original axes indices we also find which eigenvector corresponds to shortest/longest axis - that's why we use pair where first element is the length of the axis and the second one is index of the eigenvalue.
    //After sorting we can track back which eigenvector belongs to hosrtest/longest eigenvalue

    OrientationVectorsMitosis orientationVectorsMitosis;
    orientationVectorsMitosis.semiminorVec = eigenvectors[sortedAxes[0].second];
    orientationVectorsMitosis.semimajorVec = eigenvectors[sortedAxes[2].second];


    return orientationVectorsMitosis;

}


bool MitosisSimplePlugin::doDirectionalMitosis3D() {
    int currentWorkNodeNumber = pUtils->getCurrentWorkNodeNumber();

    // simplifying access to vectorized class variables
    short &split = splitVec[currentWorkNodeNumber];
    short &on = onVec[currentWorkNodeNumber];
    CellG *&childCell = childCellVec[currentWorkNodeNumber];
    CellG *&parentCell = parentCellVec[currentWorkNodeNumber];
    Point3D &splitPt = splitPtVec[currentWorkNodeNumber];

    //this implementation is valid in 3D only
    if (split && on) {
        split = false;

        WatchableField3D < CellG * > *cellField = (WatchableField3D < CellG * > *)
        potts->getCellFieldG();
        //reseting poiters to parent and child cell - neessary otherwise may get some strange side effects when mitisis is aborted
        childCell = 0;
        parentCell = 0;

        CellG *cell = cellField->get(splitPt);//cells that is being divided
        parentCell = cell;

        double xcm = cell->xCM / (float) cell->volume;
        double ycm = cell->yCM / (float) cell->volume;
        double zcm = cell->zCM / (float) cell->volume;

        set <PixelTrackerData> cellPixels = pixelTrackerAccessorPtr->get(cell->extraAttribPtr)->pixelSet;

		OrientationVectorsMitosis orientationVectorsMitosis=getOrientationVectorsMitosis3D(cell);
		
		// to divide cell along semiminor axis we take eigenvector corresponging to semimajor axis and this vector is
		// perpendicular to the division plane
		// to divide cell along semimajor axis we take eigenvector corresponging to semiminor axis and this vector is
		// perpendicular to the division plane


        //plane equation is of the form (r-p)*n=0 where p is vector pointing to point through which the plane will pass (COM)
        // n is a normal vector to the plane
        // r is (x,y,z) vector
        //nx*x+ny*y+nz*z-p*n=0
        //or nx*x+ny*y+nz*z+d=0 where d is a scalar product -p*n
        Coordinates3D<double> pVec(xcm, ycm, zcm);
        Coordinates3D<double> nVec;
        double d;

	if(divideAlongMajorAxisFlag){

		nVec=orientationVectorsMitosis.semiminorVec;;
		d=-(pVec*nVec);

	} else{
		nVec=orientationVectorsMitosis.semimajorVec;;
		d=-(pVec*nVec);
	}
			int parentCellVolume=0;
			for(set<PixelTrackerData>::iterator sitr=cellPixels.begin() ; sitr != cellPixels.end() ;++sitr){
				Coordinates3D<double> pixelTrans= boundaryStrategy->calculatePointCoordinates(sitr->pixel);
            if (nVec.x * pixelTrans.x + nVec.y * pixelTrans.y + nVec.z * pixelTrans.z + d <= 0.0) {

                if (!childCell) {
                    childCell = potts->createCellG(sitr->pixel);
					}else{
                    cellField->set(sitr->pixel, childCell);
                }
            } else {

                parentCellVolume++;
            }

			}

        if (childCell)
            return true;
        else
            return false;

    }
}


bool MitosisSimplePlugin::doDirectionalMitosisOrientationVectorBased(double _nx, double _ny, double _nz) {
    //will do directional mitosis using division axis/plane passing through center of mass and perpendicular to
    // vector (_nx,_ny,_nz)

    int currentWorkNodeNumber = pUtils->getCurrentWorkNodeNumber();

    // simplifying access to vectorized class variables
    short &split = splitVec[currentWorkNodeNumber];
    short &on = onVec[currentWorkNodeNumber];
    CellG *&childCell = childCellVec[currentWorkNodeNumber];
    CellG *&parentCell = parentCellVec[currentWorkNodeNumber];
    Point3D &splitPt = splitPtVec[currentWorkNodeNumber];

    if (!_nx && !_ny && !_nz) {
        return false; //orientation vector is 0
    }
    Coordinates3D<double> nVec(_nx, _ny, _nz);
    double norm = sqrt(nVec * nVec);
    nVec.x /= norm;
    nVec.y /= norm;
    nVec.z /= norm;


    if (split && on) {
        split = false;

        WatchableField3D < CellG * > *cellField = (WatchableField3D < CellG * > *)
        potts->getCellFieldG();
        //reseting poiters to parent and child cell - neessary otherwise may get some strange side effects when mitisis is aborted
        childCell = 0;
        parentCell = 0;

        CellG *cell = cellField->get(splitPt);//cells that is being divided

        set <PixelTrackerData> cellPixels = pixelTrackerAccessorPtr->get(cell->extraAttribPtr)->pixelSet;
        parentCell = cell;
        double xcm=cell->xCM/(float)cell->volume;
		double ycm=cell->yCM/(float)cell->volume;
		double zcm=cell->zCM/(float)cell->volume;

        //first calculate and diagonalize inertia tensor

        //plane/line equation is of the form (r-p)*n=0 where p is vector pointing to point through which the plane will pass (COM)
        // n is a normal vector to the plane/line
        // r is (x,y,z) vector
        //nx*x+ny*y+nz*z-p*n=0
        //or nx*x+ny*y+nz*z+d=0 where d is a scalar product -p*n
        Coordinates3D<double> pVec(xcm, ycm, zcm);
        double d = -(pVec * nVec);

        int parentCellVolume=0;
			for(set<PixelTrackerData>::iterator sitr=cellPixels.begin() ; sitr != cellPixels.end() ;++sitr){
				Coordinates3D<double> pixelTrans= boundaryStrategy->calculatePointCoordinates(sitr->pixel);
            if (nVec.x * pixelTrans.x + nVec.y * pixelTrans.y + nVec.z * pixelTrans.z + d <= 0.0) {

                if (!childCell) {
                    childCell = potts->createCellG(sitr->pixel);
					}else{
                    cellField->set(sitr->pixel, childCell);
                }
            } else {

                parentCellVolume++;
            }

			}

        if (childCell)
            return true;
        else
            return false;

    }
}


//bool MitosisSimplePlugin::doDirectionalMitosis3D(){
//
//	//this implementation is valid in 3D only
//	if (split && on) {
//		split = false;
//
//		WatchableField3D<CellG *> *cellField =(WatchableField3D<CellG *> *) potts->getCellFieldG();
//		//reseting poiters to parent and child cell - neessary otherwise may get some strange side effects when mitisis is aborted
//		childCell=0;
//		parentCell=0;
//
//		CellG *cell = cellField->get(splitPt);//cells that is being divided
//		parentCell=cell;
//		double xcm=cell->xCM/(float)cell->volume;
//		double ycm=cell->yCM/(float)cell->volume;
//		double zcm=cell->zCM/(float)cell->volume;
//
//		//first calculate and diagonalize inertia tensor
//		vector<vector<double> > inertiaTensor(3,vector<double>(3,0.0));
//
//		set<PixelTrackerData> cellPixels=pixelTrackerAccessorPtr->get(cell->extraAttribPtr)->pixelSet;
//		for(set<PixelTrackerData>::iterator sitr=cellPixels.begin() ; sitr != cellPixels.end() ;++sitr){
//			inertiaTensor[0][0]+=(sitr->pixel.y-ycm)*(sitr->pixel.y-ycm)+(sitr->pixel.z-zcm)*(sitr->pixel.z-zcm);
//			inertiaTensor[0][1]+=-(sitr->pixel.x-xcm)*(sitr->pixel.y-ycm);
//			inertiaTensor[0][2]+=-(sitr->pixel.x-xcm)*(sitr->pixel.z-zcm);
//			inertiaTensor[1][1]+=(sitr->pixel.x-xcm)*(sitr->pixel.x-xcm)+(sitr->pixel.z-zcm)*(sitr->pixel.z-zcm);
//			inertiaTensor[1][2]+=-(sitr->pixel.y-ycm)*(sitr->pixel.z-zcm);
//			inertiaTensor[2][2]+=(sitr->pixel.x-xcm)*(sitr->pixel.x-xcm)+(sitr->pixel.y-ycm)*(sitr->pixel.y-ycm);
//		}
//		inertiaTensor[1][0]=inertiaTensor[0][1];
//		inertiaTensor[2][0]=inertiaTensor[0][2];
//		inertiaTensor[2][1]=inertiaTensor[1][2];
//		
//	 //Finding eigenvalues
//	 vector<double> aCoeff(4,0.0);
//	 vector<complex<double> > roots;
//	 
//	 //initialize coefficients of cubic eq used to find eigenvalues of inertia tensor - before pixel copy
//	 aCoeff[0]=-1.0;
//
//	 aCoeff[1]=inertiaTensor[0][0] + inertiaTensor[1][1] + inertiaTensor[2][2];
//
//	 aCoeff[2]=inertiaTensor[0][1]*inertiaTensor[0][1] + inertiaTensor[0][2]*inertiaTensor[0][2] + inertiaTensor[1][2]*inertiaTensor[1][2]
//	 -inertiaTensor[0][0]*inertiaTensor[1][1] - inertiaTensor[0][0]*inertiaTensor[2][2] - inertiaTensor[1][1]*inertiaTensor[2][2];
//
//	 aCoeff[3]=inertiaTensor[0][0]*inertiaTensor[1][1]*inertiaTensor[2][2] + 2*inertiaTensor[0][1]*inertiaTensor[0][2]*inertiaTensor[1][2]
//	 -inertiaTensor[0][0]*inertiaTensor[1][2]*inertiaTensor[1][2]
//	 -inertiaTensor[1][1]*inertiaTensor[0][2]*inertiaTensor[0][2]
//	 -inertiaTensor[2][2]*inertiaTensor[0][1]*inertiaTensor[0][1];
//
//	 roots=solveCubicEquationRealCoeeficients(aCoeff);
//	 
//	 
//
//	 vector<Coordinates3D<double> > eigenvectors(3);
//
//	 for (int i = 0 ; i < 3 ; ++i){
//		 eigenvectors[i].x=(inertiaTensor[0][2]*(inertiaTensor[1][1]-roots[i].real())-inertiaTensor[1][2]*inertiaTensor[0][1])/
//		 (inertiaTensor[1][2]*(inertiaTensor[0][0]-roots[i].real())-inertiaTensor[0][1]*inertiaTensor[0][2]);
//		 eigenvectors[i].y=1.0;
//		 eigenvectors[i].z = (inertiaTensor[0][2]*eigenvectors[i].x+inertiaTensor[1][2]*eigenvectors[i].y)/
//		 (roots[i].real()-inertiaTensor[2][2]) ;
//		 if(eigenvectors[i].x!=eigenvectors[i].x || eigenvectors[i].y!=eigenvectors[i].y || eigenvectors[i].z!=eigenvectors[i].z){
//			return false;//simply dont do mitosis if any of the eigenvector component is NaN
//		 }
//	 }
//
//
//
//	 //finding semiaxes of the ellipsoid
//	 //Ixx=m/5.0*(a_y^2+a_z^2) - andy cyclical permutations for other coordinate combinations
//	 //a_x,a_y,a_z are lengths of semiaxes of the allipsoid
//	 // We can invert above system of equations to get:
//	vector<double> axes(3,0.0);
//
//	axes[0]=sqrt((2.5/cell->volume)*(roots[1].real()+roots[2].real()-roots[0].real()));//corresponds to first eigenvalue
//	axes[1]=sqrt((2.5/cell->volume)*(roots[0].real()+roots[2].real()-roots[1].real()));//corresponds to second eigenvalue
//	axes[2]=sqrt((2.5/cell->volume)*(roots[0].real()+roots[1].real()-roots[2].real()));//corresponds to third eigenvalue
//
//	vector<pair<double, int> > sortedAxes(3);
//	sortedAxes[0]=make_pair(axes[0],0);
//	sortedAxes[1]=make_pair(axes[1],1);
//	sortedAxes[2]=make_pair(axes[2],2);
//
//	//sorting semiaxes according the their lengths (shortest first)
//	//sort(axes.begin(),axes.end());
//	sort(sortedAxes.begin(),sortedAxes.end()); //by keeping track of original axes indices we also find which eigenvector corresponds to shortest/longest axis - that's why we use pair where first element is the length of the axis and the second one is index of the eigenvalue. 
//															 //After sorting we can track back which eigenvector belongs to hosrtest/longest eigenvalue
//

//		// to divide cell along semiminor axis we take eigenvector corresponging to semimajor axis and this vector is 
//		// perpendicular to the division plane
//		// to divide cell along semimajor axis we take eigenvector corresponging to semiminor axis and this vector is 
//		// perpendicular to the division plane
//
//
//	//plane equation is of the form (r-p)*n=0 where p is vector pointing to point through which the plane will pass (COM)
//	// n is a normal vector to the plane
//   // r is (x,y,z) vector
//	//nx*x+ny*y+nz*z-p*n=0 
//	//or nx*x+ny*y+nz*z+d=0 where d is a scalar product -p*n
//	Coordinates3D<double> pVec(xcm,ycm,zcm);
//	Coordinates3D<double> nVec;
//	double d;
//
//	if(divideAlongMajorAxisFlag){
//	
//		nVec=eigenvectors[sortedAxes[0].second];
//		d=-(pVec*nVec);
//		
//	} else{
//		nVec=eigenvectors[sortedAxes[2].second];
//		d=-(pVec*nVec);
//	}
//
//			int parentCellVolume=0;
//			for(set<PixelTrackerData>::iterator sitr=cellPixels.begin() ; sitr != cellPixels.end() ;++sitr){
//				if(nVec.x*sitr->pixel.x+nVec.y*sitr->pixel.y + nVec.z*sitr->pixel.z+d<= 0.0){
//					
//					if(!childCell){
//						childCell = potts->createCellG(sitr->pixel);
//					}else{
//						cellField->set(sitr->pixel, childCell);
//					}
//				}else{
//					
//					parentCellVolume++;
//				}
//
//			}
//
//		if(childCell)
//			return true;
//		else
//			return false;
//
//	}
//}


//bool MitosisSimplePlugin::doDirectionalMitosis2D_xy(){
//	//this implementation is valid in 2D only
//	if (split && on) {
//		split = false;
//
//		WatchableField3D<CellG *> *cellField =(WatchableField3D<CellG *> *) potts->getCellFieldG();
//		//reseting poiters to parent and child cell - neessary otherwise may get some strange side effects when mitisis is aborted
//		childCell=0;
//		parentCell=0;
//
//		CellG *cell = cellField->get(splitPt);//cells that is being divided
//		parentCell=cell;
//		double xcm=cell->xCM/(float)cell->volume;
//		double ycm=cell->yCM/(float)cell->volume;
//		double zcm=cell->zCM/(float)cell->volume;
//
//		//first calculate and diagonalize inertia tensor
//		vector<vector<double> > inertiaTensor(2,vector<double>(2,0.0));
//
//		set<PixelTrackerData> cellPixels=pixelTrackerAccessorPtr->get(cell->extraAttribPtr)->pixelSet;
//		for(set<PixelTrackerData>::iterator sitr=cellPixels.begin() ; sitr != cellPixels.end() ;++sitr){
//			inertiaTensor[0][0]+=(sitr->pixel.y-ycm)*(sitr->pixel.y-ycm);
//			inertiaTensor[0][1]+=-(sitr->pixel.x-xcm)*(sitr->pixel.y-ycm);
//			inertiaTensor[1][1]+=(sitr->pixel.x-xcm)*(sitr->pixel.x-xcm);
//		}
//		inertiaTensor[1][0]=inertiaTensor[0][1];
//
//		double radical=0.5*sqrt((inertiaTensor[0][0]-inertiaTensor[1][1])*(inertiaTensor[0][0]-inertiaTensor[1][1])+4.0*inertiaTensor[0][1]*inertiaTensor[0][1]);			
//		double lMin=0.5*(inertiaTensor[0][0]+inertiaTensor[1][1])-radical;
//		double lMax=0.5*(inertiaTensor[0][0]+inertiaTensor[1][1])+radical;
//
//		//orientationVec points along semiminor axis (it corresponds to larger eigenvalue)
//		Coordinates3D<double> orientationVec;
//		if(inertiaTensor[0][1]!=0.0){
//
//			orientationVec=Coordinates3D<double>(inertiaTensor[0][1],lMax-inertiaTensor[0][0],0.0);
//			double length=sqrt(orientationVec.x*orientationVec.x+orientationVec.y*orientationVec.y+orientationVec.z*orientationVec.z);
//			orientationVec.x/=length;
//			orientationVec.y/=length;
//			orientationVec.z/=length;
//		}else{
//			if(inertiaTensor[0][0]>inertiaTensor[1][1])
//				orientationVec=Coordinates3D<double>(0.0,1.0,0.0);
//			else
//				orientationVec=Coordinates3D<double>(1.0,0.0,0.0);
//		}
//
//
//		//once we know orientation vector corresponding to bigger eigenvalue (pointing along semiminor axis) we may divide cell 
//		//along major or minor axis
//
//		// assume the following form of the equation of the straight line passing through COM (xcm,ycm,zcm) of cell being divided
//		//y=a*x+b;
//		double a;
//		double b;
//		//determining coefficients of the straight line passing through
//		if(divideAlongMinorAxisFlag){
//
//
//			a=orientationVec.y/orientationVec.x;
//			b=ycm-xcm*a;
//
//			if(a!=a){//a is Nan - will happen when orientationVec.x is 0.0 thus minor axis is along y axis
//				a=0.0;
//				b=0.0;
//			}
//		}
//
//		if(divideAlongMajorAxisFlag){
//			if(orientationVec.y==0.0){//then perpendicular vector (major axis) is along y axis meaning:
//				a=0.0;
//				b=0.0;
//
//			} else{
//				a=-orientationVec.x/orientationVec.y;
//				b=ycm-xcm*a;
//			}
//		}
//
//		//now do the division
//
//		if(a==0.0 && b==0.0){//division along y axis
//
//			for(set<PixelTrackerData>::iterator sitr=cellPixels.begin() ; sitr != cellPixels.end() ;++sitr){
//				if(sitr->pixel.x<xcm){
//					if(!childCell){
//						childCell = potts->createCellG(sitr->pixel);
//					}else{
//						cellField->set(sitr->pixel, childCell);
//					}
//				}
//			}
//
//		}else{//division will be done along axis different than y axis
//			int parentCellVolume=0;
//			for(set<PixelTrackerData>::iterator sitr=cellPixels.begin() ; sitr != cellPixels.end() ;++sitr){
//				if(sitr->pixel.y <= a*sitr->pixel.x+b){
//					
//					if(!childCell){
//						childCell = potts->createCellG(sitr->pixel);
//					}else{
//						cellField->set(sitr->pixel, childCell);
//					}
//				}else{
//					
//					parentCellVolume++;
//				}
//
//			}
//		}
//
//		//if childCell was created this means mitosis was sucessful. If child cell was not created there was no mitosis
//		if(childCell)
//			return true;
//		else
//			return false;
//	}
//}
