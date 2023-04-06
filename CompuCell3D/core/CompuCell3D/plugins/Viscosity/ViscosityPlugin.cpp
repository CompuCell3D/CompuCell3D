#include <CompuCell3D/CC3D.h>

#include "ViscosityPlugin.h"

using namespace CompuCell3D;

#include <CompuCell3D/plugins/NeighborTracker/NeighborTrackerPlugin.h>
#include <Logger/CC3DLogger.h>

ViscosityPlugin::ViscosityPlugin() : potts(0), sim(0), neighborTrackerAccessorPtr(0), lambdaViscosity(0),
                                     maxNeighborIndex(0) {
}

ViscosityPlugin::~ViscosityPlugin() {
}

double ViscosityPlugin::dist(double _x, double _y, double _z) {
    return sqrt(_x * _x + _y * _y + _z * _z);
}

void ViscosityPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
    potts = simulator->getPotts();
    sim = simulator;

    bool pluginAlreadyRegisteredFlagCOM;
    //this will load CenterOFMass plugin if it is not already loaded
    Plugin *pluginCOM = Simulator::pluginManager.get("CenterOfMass",
                                                     &pluginAlreadyRegisteredFlagCOM);

    bool pluginAlreadyRegisteredFlag;
    //this will load NeighborTracker plugin if it is not already loaded
    Plugin *plugin = Simulator::pluginManager.get("NeighborTracker",
                                                  &pluginAlreadyRegisteredFlag);




    pluginName = _xmlData->getAttribute("Name");


    if (!pluginAlreadyRegisteredFlag)
        plugin->init(simulator);
    potts->registerEnergyFunctionWithName(this, toString());
    //save pointer to plugin xml element for later. Initialization has to be done in extraInit
    // to make sure automaton (CelltypePlugin)
    // is already registered - we need it in the case of BYCELLTYPE
    xmlData = _xmlData;

    simulator->registerSteerableObject(this);


    boundaryStrategy = BoundaryStrategy::getInstance();
    potts->getBoundaryXName() == "Periodic" ? boundaryConditionIndicator.x = 1 : boundaryConditionIndicator.x = 0;
    potts->getBoundaryYName() == "Periodic" ? boundaryConditionIndicator.y = 1 : boundaryConditionIndicator.y = 0;
    potts->getBoundaryZName() == "Periodic" ? boundaryConditionIndicator.z = 1 : boundaryConditionIndicator.z = 0;

    fieldDim = potts->getCellFieldG()->getDim();

    maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);
}

void ViscosityPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {

    lambdaViscosity = _xmlData->getFirstElement("LambdaViscosity")->getDouble();

}

void ViscosityPlugin::extraInit(Simulator *simulator) {

    update(xmlData);
    bool pluginAlreadyRegisteredFlag;
    NeighborTrackerPlugin *nTrackerPlugin = (NeighborTrackerPlugin *) Simulator::pluginManager.get("NeighborTracker",
                                                                                                   &pluginAlreadyRegisteredFlag); //this will load VolumeTracker plugin if it is not already loaded

    neighborTrackerAccessorPtr = nTrackerPlugin->getNeighborTrackerAccessorPtr();

    //viscosityEnergy->initializeViscosityEnergy();
}

double ViscosityPlugin::changeEnergy(const Point3D &pt, const CellG *newCell, const CellG *oldCell) {
    if (sim->getStep() < 100) {
        return 0;
    }

    double energy = 0;
    unsigned int token = 0;
    double distance = 0;
    Point3D n;
    double cellDistance = 0.0;
    double commonArea = 0.0;
    double x0, y0, z0, x1, y1, z1;
    double velocityDiffX = 0;
    double velocityDiffY = 0;
    double velocityDiffZ = 0;
    Coordinates3D<double> nCellCom0, nCellCom1, cellCom0, cellCom1;

    Coordinates3D<double> oldCellCMBefore, oldCellCMBeforeBefore, oldCellCMAfter, newCellCMBefore, newCellCMBeforeBefore, newCellCMAfter;
    Coordinates3D<double> nCellCMBefore, nCellCMBeforeBefore, nCellCMAfter;

    Coordinates3D<double> oldCellVel, nCellVel, newCellVel, distanceInvariantVec;

    CellG *nCell = 0;
    WatchableField3D < CellG * > *fieldG = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();

    std::set <NeighborSurfaceData> *oldCellNeighborsPtr = 0;
    std::set <NeighborSurfaceData> *newCellNeighborsPtr = 0;

    std::set<NeighborSurfaceData>::iterator sitr;

    set <NeighborSurfaceData> oldCellPixelNeighborSurfaceData;
    set<NeighborSurfaceData>::iterator sitrNSD;
    set<NeighborSurfaceData>::iterator sitrNSDTmp;

    bool printFlag = false;


    if (oldCell) {

        //new code
        oldCellNeighborsPtr = &(neighborTrackerAccessorPtr->get(oldCell->extraAttribPtr)->cellNeighbors);
        oldCellCMAfter = precalculateCentroid(pt, oldCell, -1, fieldDim, boundaryStrategy);

        if (oldCell->volume > 1) {
            oldCellCMAfter.x = oldCellCMAfter.x / (oldCell->volume - 1);
            oldCellCMAfter.y = oldCellCMAfter.y / (oldCell->volume - 1);
            oldCellCMAfter.z = oldCellCMAfter.z / (oldCell->volume - 1);
        } else {
            oldCellCMAfter.x = oldCellCMAfter.x / oldCell->volume;
            oldCellCMAfter.y = oldCellCMAfter.y / oldCell->volume;
            oldCellCMAfter.z = oldCellCMAfter.z / oldCell->volume;
        }

        oldCellCMBefore = Coordinates3D<double>(
                oldCell->xCM / (double) oldCell->volume,
                oldCell->yCM / (double) oldCell->volume,
                oldCell->zCM / (double) oldCell->volume

        );

        oldCellCMBeforeBefore = Coordinates3D<double>(oldCell->xCOMPrev, oldCell->yCOMPrev, oldCell->zCOMPrev);

        //new code


    }
    if (newCell) {

        //new code
        newCellNeighborsPtr = &(neighborTrackerAccessorPtr->get(newCell->extraAttribPtr)->cellNeighbors);
        newCellCMAfter = precalculateCentroid(pt, newCell, +1, fieldDim, boundaryStrategy);

        newCellCMAfter.x = newCellCMAfter.x / (newCell->volume + 1);
        newCellCMAfter.y = newCellCMAfter.y / (newCell->volume + 1);
        newCellCMAfter.z = newCellCMAfter.z / (newCell->volume + 1);

        newCellCMBefore = Coordinates3D<double>(newCell->xCM / (double) newCell->volume,
                                                newCell->yCM / (double) newCell->volume,
                                                newCell->zCM / (double) newCell->volume);


        newCellCMBeforeBefore = Coordinates3D<double>(newCell->xCOMPrev, newCell->yCOMPrev, newCell->zCOMPrev);
    }

    //will compute here common surface area of old cell pixel with its all nearest neighbors

    Neighbor neighbor;
    //will compute here common surface area of old cell pixel with its all nearest neighbors
    for (unsigned int nIdx = 0; nIdx <= maxNeighborIndex; ++nIdx) {
        neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D &>(pt), nIdx);
        if (!neighbor.distance) {
            //if distance is 0 then the neighbor returned is invalid
            continue;
        }

        nCell = fieldG->get(neighbor.pt);
        if (!nCell) continue;
        sitrNSD = oldCellPixelNeighborSurfaceData.find(NeighborSurfaceData(nCell));
        if (sitrNSD != oldCellPixelNeighborSurfaceData.end()) {
            sitrNSD->incrementCommonSurfaceArea(*sitrNSD);
        } else {
            oldCellPixelNeighborSurfaceData.insert(NeighborSurfaceData(nCell, 1));
        }
    }

    //NOTE: There is a double counting issue count energy between old and new cell - as it is written in the paper
    //energy before flip from old cell
    if (oldCell) {


        for (sitr = oldCellNeighborsPtr->begin(); sitr != oldCellNeighborsPtr->end(); ++sitr) {

            nCell = sitr->neighborAddress;

            if (!nCell) continue; //in case medium is a neeighbor

            commonArea = sitr->commonSurfaceArea;


            //cell velocity data -  difference!

            nCellCMBefore = Coordinates3D<double>(
                    nCell->xCM / (double) nCell->volume,
                    nCell->yCM / (double) nCell->volume,
                    nCell->zCM / (double) nCell->volume

            );

            nCellCMBeforeBefore = Coordinates3D<double>(nCell->xCOMPrev, nCell->yCOMPrev, nCell->zCOMPrev);

            oldCellVel = distanceVectorCoordinatesInvariant(oldCellCMBefore, oldCellCMBeforeBefore, fieldDim);
            nCellVel = distanceVectorCoordinatesInvariant(nCellCMBefore, nCellCMBeforeBefore, fieldDim);

            velocityDiffX = oldCellVel.x - nCellVel.x;
            velocityDiffY = oldCellVel.y - nCellVel.y;
            velocityDiffZ = oldCellVel.z - nCellVel.z;


            distanceInvariantVec = distanceVectorCoordinatesInvariant(oldCellCMBefore, nCellCMBefore, fieldDim);

            x0 = distanceInvariantVec.x;
            y0 = distanceInvariantVec.y;
            z0 = distanceInvariantVec.z;

            cellDistance = dist(x0, y0, z0);



            if (nCell == newCell) {
                energy -= commonArea * (
                        velocityDiffX * velocityDiffX * sqrt((y0) * (y0) + (z0) * (z0))
                        + velocityDiffY * velocityDiffY * sqrt((z0) * (z0) + (x0) * (x0))
                        + velocityDiffZ * velocityDiffZ * sqrt((x0) * (x0) + (y0) * (y0))
                )
                          / (cellDistance * cellDistance * cellDistance);
            } else {
                energy -= commonArea * (
                        velocityDiffX * velocityDiffX * sqrt((y0) * (y0) + (z0) * (z0))
                        + velocityDiffY * velocityDiffY * sqrt((z0) * (z0) + (x0) * (x0))
                        + velocityDiffZ * velocityDiffZ * sqrt((x0) * (x0) + (y0) * (y0))
                )
                          / (cellDistance * cellDistance * cellDistance);

            }


        }


    }


    //energy before flip from new cell
    if (newCell) {


        for (sitr = newCellNeighborsPtr->begin(); sitr != newCellNeighborsPtr->end(); ++sitr) {

            nCell = sitr->neighborAddress;

            if (!nCell) continue; //in case medium is a nieighbor
            ///DOUBLE COUNTING PROTECTION *******************************************************************

            if (nCell == oldCell) continue; //to avoid double counting of newCell-oldCell energy


            commonArea = sitr->commonSurfaceArea;


            //cell velocity data -  difference!

            nCellCMBefore = Coordinates3D<double>(
                    nCell->xCM / (double) nCell->volume,
                    nCell->yCM / (double) nCell->volume,
                    nCell->zCM / (double) nCell->volume

            );

            nCellCMBeforeBefore = Coordinates3D<double>(nCell->xCOMPrev, nCell->yCOMPrev, nCell->zCOMPrev);

            newCellVel = distanceVectorCoordinatesInvariant(newCellCMBefore, newCellCMBeforeBefore, fieldDim);
            nCellVel = distanceVectorCoordinatesInvariant(nCellCMBefore, nCellCMBeforeBefore, fieldDim);

            velocityDiffX = newCellVel.x - nCellVel.x;
            velocityDiffY = newCellVel.y - nCellVel.y;
            velocityDiffZ = newCellVel.z - nCellVel.z;


            distanceInvariantVec = distanceVectorCoordinatesInvariant(newCellCMBefore, nCellCMBefore, fieldDim);

            x0 = distanceInvariantVec.x;
            y0 = distanceInvariantVec.y;
            z0 = distanceInvariantVec.z;

            cellDistance = dist(x0, y0, z0);


            energy -= commonArea * (
                    velocityDiffX * velocityDiffX * sqrt((y0) * (y0) + (z0) * (z0))
                    + velocityDiffY * velocityDiffY * sqrt((z0) * (z0) + (x0) * (x0))
                    + velocityDiffZ * velocityDiffZ * sqrt((x0) * (x0) + (y0) * (y0))
            )
                      / (cellDistance * cellDistance * cellDistance);


        }

    }


    //energy after flip from old cell
    //NOTE:old cell can only loose neighbors with one exception - when new and old cells do not touch each other before pixel copy then old cell will gain new neighbor (so will newCell).
    //In such a case we still do calculations in the last section which handles new neighbors of newCell.
    if (oldCell) {

        for (sitr = oldCellNeighborsPtr->begin(); sitr != oldCellNeighborsPtr->end(); ++sitr) {
            nCell = sitr->neighborAddress;
            if (!nCell) continue; //in case medium is a neighbor
            //will need to adjust commonArea for after flip case
            commonArea = sitr->commonSurfaceArea;
            sitrNSD = oldCellPixelNeighborSurfaceData.find(NeighborSurfaceData(nCell));

            if (sitrNSD != oldCellPixelNeighborSurfaceData.end()) {
                if (sitrNSD->neighborAddress !=
                    newCell) {
                    // if neighbor pixel is not a newCell we decrement commonArea by the oldCellPixelNeighborSurface
                    commonArea -= sitrNSD->commonSurfaceArea;
                } else {//otherwise we do the following
                    sitrNSDTmp = oldCellPixelNeighborSurfaceData.find(
                            NeighborSurfaceData(const_cast<CellG *>(oldCell)));
                    commonArea -= sitrNSD->commonSurfaceArea;//we subtract common area of pixel with newCell
                    if (sitrNSDTmp != oldCellPixelNeighborSurfaceData.end()) {// in case old cell is not
                        //on the list of oldPixelNeighbors
                        commonArea += sitrNSDTmp->commonSurfaceArea;//we add common area of pixel with oldCell
                    }

                }
            }



            if (commonArea < 0.0) { //just in case
                commonArea = 0.0;
                CC3D_Log(LOG_DEBUG) << "reached below zero old after";			}
			if(nCell!=newCell){


                nCellCMBefore = Coordinates3D<double>(
                        nCell->xCM / (double) nCell->volume,
                        nCell->yCM / (double) nCell->volume,
                        nCell->zCM / (double) nCell->volume
                );

                nCellCMBeforeBefore = Coordinates3D<double>(nCell->xCOMPrev, nCell->yCOMPrev, nCell->zCOMPrev);


                oldCellVel = distanceVectorCoordinatesInvariant(oldCellCMAfter, oldCellCMBefore, fieldDim);
                nCellVel = distanceVectorCoordinatesInvariant(nCellCMBefore, nCellCMBeforeBefore,
                                                              fieldDim); //if nCell is not a new cell then its velocity before and after spin flip is the same - so I am using earlier expression

                velocityDiffX = oldCellVel.x - nCellVel.x;
                velocityDiffY = oldCellVel.y - nCellVel.y;
                velocityDiffZ = oldCellVel.z - nCellVel.z;


                nCellCMAfter = Coordinates3D<double>(
                        nCell->xCM / (double) nCell->volume,
                        nCell->yCM / (double) nCell->volume,
                        nCell->zCM / (double) nCell->volume

                );

            } else {
                oldCellVel = distanceVectorCoordinatesInvariant(oldCellCMAfter, oldCellCMBefore, fieldDim);
                nCellVel = distanceVectorCoordinatesInvariant(newCellCMAfter, newCellCMBefore, fieldDim);


                velocityDiffX = oldCellVel.x - nCellVel.x;
                velocityDiffY = oldCellVel.y - nCellVel.y;
                velocityDiffZ = oldCellVel.z - nCellVel.z;

                nCellCMAfter = newCellCMAfter;
            }

            distanceInvariantVec = distanceVectorCoordinatesInvariant(oldCellCMAfter, nCellCMAfter, fieldDim);

            x0 = distanceInvariantVec.x;
            y0 = distanceInvariantVec.y;
            z0 = distanceInvariantVec.z;

            cellDistance = dist(x0, y0, z0);




            if (nCell == newCell) {
                energy += commonArea * (
                        velocityDiffX * velocityDiffX * sqrt((y0) * (y0) + (z0) * (z0))
                        + velocityDiffY * velocityDiffY * sqrt((z0) * (z0) + (x0) * (x0))
                        + velocityDiffZ * velocityDiffZ * sqrt((x0) * (x0) + (y0) * (y0))
                )
                          / (cellDistance * cellDistance * cellDistance);
            } else {
                energy += commonArea * (
                        velocityDiffX * velocityDiffX * sqrt((y0) * (y0) + (z0) * (z0))
                        + velocityDiffY * velocityDiffY * sqrt((z0) * (z0) + (x0) * (x0))
                        + velocityDiffZ * velocityDiffZ * sqrt((x0) * (x0) + (y0) * (y0))
                )
                          / (cellDistance * cellDistance * cellDistance);

            }

        }
    }


    //energy after flip from new cell
    if (newCell) {

        for (sitr = newCellNeighborsPtr->begin(); sitr != newCellNeighborsPtr->end(); ++sitr) {
            nCell = sitr->neighborAddress;
            if (!nCell) continue; //in case medium is a nieighbor

            ///DOUBLE COUNTING PROTECTION *******************************************************************
            if (nCell == oldCell) continue; //to avoid double counting of newCell-oldCell eenrgy
            //will need to adjust commonArea for after flip case
            commonArea = sitr->commonSurfaceArea;

            sitrNSD = oldCellPixelNeighborSurfaceData.find(NeighborSurfaceData(nCell));
            if (sitrNSD != oldCellPixelNeighborSurfaceData.end()) {
                if (sitrNSD->neighborAddress != oldCell) { // if neighbor is not a oldCell we increment commonArea
                    commonArea += sitrNSD->commonSurfaceArea;
                } else {//otherwise we do the following
                    sitrNSDTmp = oldCellPixelNeighborSurfaceData.find(
                            NeighborSurfaceData(const_cast<CellG *>(newCell)));
                    if (sitrNSDTmp != oldCellPixelNeighborSurfaceData.end()) {// in case new cell is not
                        //on the list of oldPixelNeighbors
                        commonArea -= sitrNSDTmp->commonSurfaceArea;//we subtract common area of pixel with newCell
                    }
                    commonArea += sitrNSD->commonSurfaceArea;//we add common area of pixel with oldCell

                }
            }
            if (commonArea < 0.0) { //just in case
                commonArea = 0.0;
                CC3D_Log(LOG_DEBUG) << "reached below zero new after";
            }


            if (nCell != oldCell) {


                nCellCMBefore = Coordinates3D<double>(
                        nCell->xCM / (double) nCell->volume,
                        nCell->yCM / (double) nCell->volume,
                        nCell->zCM / (double) nCell->volume
                );

                nCellCMBeforeBefore = Coordinates3D<double>(nCell->xCOMPrev, nCell->yCOMPrev, nCell->zCOMPrev);

                newCellVel = distanceVectorCoordinatesInvariant(newCellCMAfter, newCellCMBefore, fieldDim);
                //if nCell is not an old cell then its velocity before and after spin flip is the same -
                // so I am using earlier expression
                nCellVel = distanceVectorCoordinatesInvariant(nCellCMBefore, nCellCMBeforeBefore,
                                                              fieldDim);

                velocityDiffX = newCellVel.x - nCellVel.x;
                velocityDiffY = newCellVel.y - nCellVel.y;
                velocityDiffZ = newCellVel.z - nCellVel.z;

                nCellCMAfter = Coordinates3D<double>(
                        nCell->xCM / (double) nCell->volume,
                        nCell->yCM / (double) nCell->volume,
                        nCell->zCM / (double) nCell->volume
                );


            } else {
                //this should never get executed
                newCellVel = distanceVectorCoordinatesInvariant(newCellCMAfter, newCellCMBefore, fieldDim);
                nCellVel = distanceVectorCoordinatesInvariant(oldCellCMAfter, oldCellCMBefore, fieldDim);


                velocityDiffX = oldCellVel.x - nCellVel.x;
                velocityDiffY = oldCellVel.y - nCellVel.y;
                velocityDiffZ = oldCellVel.z - nCellVel.z;

                nCellCMAfter = oldCellCMAfter;
                CC3D_Log(LOG_DEBUG) << "EXECUTING FORBIDDEN CODE";


            }


            distanceInvariantVec = distanceVectorCoordinatesInvariant(newCellCMAfter, nCellCMAfter, fieldDim);

            x0 = distanceInvariantVec.x;
            y0 = distanceInvariantVec.y;
            z0 = distanceInvariantVec.z;

            cellDistance = dist(x0, y0, z0);


            energy += commonArea * (
                    velocityDiffX * velocityDiffX * sqrt((y0) * (y0) + (z0) * (z0))
                    + velocityDiffY * velocityDiffY * sqrt((z0) * (z0) + (x0) * (x0))
                    + velocityDiffZ * velocityDiffZ * sqrt((x0) * (x0) + (y0) * (y0))
            )
                      / (cellDistance * cellDistance * cellDistance);


        }


    }


    return lambdaViscosity * energy;

}


std::string ViscosityPlugin::steerableName() {
    return pluginName;
}

std::string ViscosityPlugin::toString() {
    return pluginName;
}
