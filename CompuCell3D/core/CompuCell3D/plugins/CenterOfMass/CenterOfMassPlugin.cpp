
#include <CompuCell3D/CC3D.h>

using namespace CompuCell3D;


using namespace std;

#include "CenterOfMassPlugin.h"
#include <Logger/CC3DLogger.h>

CenterOfMassPlugin::CenterOfMassPlugin() : boundaryStrategy(0) {}

CenterOfMassPlugin::~CenterOfMassPlugin() {}

void CenterOfMassPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
	boundaryStrategy=BoundaryStrategy::getInstance();
	CC3D_Log(LOG_DEBUG) << std::endl << std::endl << std::endl << "  \t\t\t CenterOfMassPlugin::init() - CALLING INIT OF CENTER OF MASS PLUGIN" << std::endl << std::endl << std::endl;
    potts = simulator->getPotts();
    bool pluginAlreadyRegisteredFlag;
    Plugin *plugin = Simulator::pluginManager.get("VolumeTracker",
                                                  &pluginAlreadyRegisteredFlag); //this will load VolumeTracker plugin if it is not already loaded
    if (!pluginAlreadyRegisteredFlag)
        plugin->init(simulator);

    potts->registerCellGChangeWatcher(this);

    potts->getBoundaryXName() == "Periodic" ? boundaryConditionIndicator.x = 1 : boundaryConditionIndicator.x = 0;
    potts->getBoundaryYName() == "Periodic" ? boundaryConditionIndicator.y = 1 : boundaryConditionIndicator.y = 0;
    potts->getBoundaryZName() == "Periodic" ? boundaryConditionIndicator.z = 1 : boundaryConditionIndicator.z = 0;

    fieldDim = potts->getCellFieldG()->getDim();

    //determining allowedAreaMin and allowedAreaMax - this seems elaborate but will work for all lattices CC3D supports
    if (boundaryStrategy->getLatticeType() == HEXAGONAL_LATTICE) {
        allowedAreaMin.x = 0.0;
        allowedAreaMin.y = (fieldDim.z >= 3 ? -sqrt(3.0) / 6.0 : 0.0);
        allowedAreaMin.z = 0.0;

        allowedAreaMax.x = fieldDim.x + 0.5;
        allowedAreaMax.y = fieldDim.y * sqrt(3.0) / 2.0 + (fieldDim.z >= 3 ? sqrt(3.0) / 6.0 : 0.0);
        allowedAreaMax.z = fieldDim.z * sqrt(6.0) / 3.0;

    } else {
        allowedAreaMin.x = 0.0;
        allowedAreaMin.y = 0.0;
        allowedAreaMin.z = 0.0;

        allowedAreaMax.x = fieldDim.x;
        allowedAreaMax.y = fieldDim.y;
        allowedAreaMax.z = fieldDim.z;

    }

}

void CompuCell3D::CenterOfMassPlugin::field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell) {

    if (newCell == oldCell) //this may happen if you are trying to assign same cell to one pixel twice
        return;

    Coordinates3D<double> ptTrans = boundaryStrategy->calculatePointCoordinates(pt);
    if (!boundaryConditionIndicator.x && !boundaryConditionIndicator.y && !boundaryConditionIndicator.z) {

        if (oldCell) {
            //temporary code to check if viscosity is working - volume tracker always runs before COM plugin
            if (!potts->checkIfFrozen(oldCell->type)) {
                oldCell->xCOMPrev = oldCell->xCM / (oldCell->volume + 1);
                oldCell->yCOMPrev = oldCell->yCM / (oldCell->volume + 1);
                oldCell->zCOMPrev = oldCell->zCM / (oldCell->volume + 1);
            }


            oldCell->xCM -= ptTrans.x;
            oldCell->yCM -= ptTrans.y;
            oldCell->zCM -= ptTrans.z;

            //storing actual center of mass
            if (oldCell->volume) {
                oldCell->xCOM = oldCell->xCM / oldCell->volume;
                oldCell->yCOM = oldCell->yCM / oldCell->volume;
                oldCell->zCOM = oldCell->zCM / oldCell->volume;
            } else {
                oldCell->xCOM = 0.0;
                oldCell->yCOM = 0.0;
                oldCell->zCOM = 0.0;
            }

            if (potts->checkIfFrozen(oldCell->type)) {
                oldCell->xCOMPrev = oldCell->xCM / (oldCell->volume);
                oldCell->yCOMPrev = oldCell->yCM / (oldCell->volume);
                oldCell->zCOMPrev = oldCell->zCM / (oldCell->volume);
            }
        }

        if (newCell) {
            //temporary code to check if viscosity is working - volume tracker always runs before COM plugin
            if (!potts->checkIfFrozen(newCell->type)) {
                if (newCell->volume > 1) {
                    newCell->xCOMPrev = newCell->xCM / (newCell->volume - 1);
                    newCell->yCOMPrev = newCell->yCM / (newCell->volume - 1);
                    newCell->zCOMPrev = newCell->zCM / (newCell->volume - 1);
                } else {
                    newCell->xCOMPrev = newCell->xCM;
                    newCell->yCOMPrev = newCell->yCM;
                    newCell->zCOMPrev = newCell->zCM;

                }
            }

            newCell->xCM += ptTrans.x;
            newCell->yCM += ptTrans.y;
            newCell->zCM += ptTrans.z;

            //storing actual center of mass
            newCell->xCOM = newCell->xCM / newCell->volume;
            newCell->yCOM = newCell->yCM / newCell->volume;
            newCell->zCOM = newCell->zCM / newCell->volume;

            if (potts->checkIfFrozen(newCell->type)) {

                newCell->xCOMPrev = newCell->xCM / (newCell->volume);
                newCell->yCOMPrev = newCell->yCM / (newCell->volume);
                newCell->zCOMPrev = newCell->zCM / (newCell->volume);

            }

        }
        return;
    }

    //if there are boundary conditions defined that we have to do some shifts to correctly calculate center of mass
    //This approach will work only for cells whose span is much smaller that lattice dimension
    // in the "periodic "direction
    //e.g. cell that is very long and "wraps lattice" will have miscalculated CM using this algorithm.
    // On the other hand, you do not really expect
    //cells to have dimensions comparable to lattice...

    if (oldCell) {
        //temporary code to check if viscosity is working - volume tracker always runs before COM plugin
        if (!potts->checkIfFrozen(oldCell->type)) {
            oldCell->xCOMPrev = oldCell->xCM / (oldCell->volume + 1);
            oldCell->yCOMPrev = oldCell->yCM / (oldCell->volume + 1);
            oldCell->zCOMPrev = oldCell->zCM / (oldCell->volume + 1);
        }


    }

    if (newCell) {
        //temporary code to check if viscosity is working - volume tracker always runs before COM plugin
        if (!potts->checkIfFrozen(newCell->type)) {
            if (newCell->volume > 1) {
                newCell->xCOMPrev = newCell->xCM / (newCell->volume - 1);
                newCell->yCOMPrev = newCell->yCM / (newCell->volume - 1);
                newCell->zCOMPrev = newCell->zCM / (newCell->volume - 1);
            } else {
                newCell->xCOMPrev = newCell->xCM;
                newCell->yCOMPrev = newCell->yCM;
                newCell->zCOMPrev = newCell->zCM;

            }
        }

    }

    Coordinates3D<double> shiftVec;
    Coordinates3D<double> shiftedPt;
    Coordinates3D<double> distanceVecMin;
    //determines minimum coordinates for the perpendicular lines passinig through pt
    Coordinates3D<double> distanceVecMax;
    Coordinates3D<double> distanceVecMax_1;
    //determines minimum coordinates for the perpendicular lines passinig through pt
    Coordinates3D<double> distanceVec; //measures lattice distances along x,y,z -
    // they can be different for different lattices. The lines have to pass through pt

    distanceVecMin.x = boundaryStrategy->calculatePointCoordinates(Point3D(0, pt.y, pt.z)).x;
    distanceVecMin.y = boundaryStrategy->calculatePointCoordinates(Point3D(pt.x, 0, pt.z)).y;
    distanceVecMin.z = boundaryStrategy->calculatePointCoordinates(Point3D(pt.x, pt.y, 0)).z;

    distanceVecMax.x = boundaryStrategy->calculatePointCoordinates(Point3D(fieldDim.x, pt.y, pt.z)).x;
    distanceVecMax.y = boundaryStrategy->calculatePointCoordinates(Point3D(pt.x, fieldDim.y, pt.z)).y;
    distanceVecMax.z = boundaryStrategy->calculatePointCoordinates(Point3D(pt.x, pt.y, fieldDim.z)).z;

    distanceVecMax_1.x = boundaryStrategy->calculatePointCoordinates(Point3D(fieldDim.x - 1, pt.y, pt.z)).x;
    distanceVecMax_1.y = boundaryStrategy->calculatePointCoordinates(Point3D(pt.x, fieldDim.y - 1, pt.z)).y;
    distanceVecMax_1.z = boundaryStrategy->calculatePointCoordinates(Point3D(pt.x, pt.y, fieldDim.z - 1)).z;

    distanceVec = distanceVecMax - distanceVecMin;

    Coordinates3D<double> fieldDimTrans = boundaryStrategy->calculatePointCoordinates(
            Point3D(fieldDim.x - 1, fieldDim.y - 1, fieldDim.z - 1));

    double xCM, yCM, zCM; //temporary centroids

    double x, y, z;
    double xo, yo, zo;

    if (oldCell) {
        xo = oldCell->xCM;
        yo = oldCell->yCM;
        zo = oldCell->zCM;

        x = oldCell->xCM - ptTrans.x;
        y = oldCell->yCM - ptTrans.y;
        z = oldCell->zCM - ptTrans.z;

        //calculating shiftVec - to translate CM



        //shift is defined to be zero vector for non-periodic b.c. - everything reduces to naive calculations then
        shiftVec.x =
                (oldCell->xCM / (oldCell->volume + 1) - ((int) fieldDimTrans.x) / 2) * boundaryConditionIndicator.x;
        shiftVec.y =
                (oldCell->yCM / (oldCell->volume + 1) - ((int) fieldDimTrans.y) / 2) * boundaryConditionIndicator.y;
        shiftVec.z =
                (oldCell->zCM / (oldCell->volume + 1) - ((int) fieldDimTrans.z) / 2) * boundaryConditionIndicator.z;

        //shift CM to approximately center of lattice, new centroids are:
        xCM = oldCell->xCM - shiftVec.x * (oldCell->volume + 1);
        yCM = oldCell->yCM - shiftVec.y * (oldCell->volume + 1);
        zCM = oldCell->zCM - shiftVec.z * (oldCell->volume + 1);
        //Now shift pt
        shiftedPt = ptTrans;
        shiftedPt -= shiftVec;

        //making sure that shifted point is in the lattice
        if (shiftedPt.x < distanceVecMin.x) {
            shiftedPt.x += distanceVec.x;
        } else if (shiftedPt.x > distanceVecMax_1.x) {
            shiftedPt.x -= distanceVec.x;
        }

        if (shiftedPt.y < distanceVecMin.y) {
            shiftedPt.y += distanceVec.y;
        } else if (shiftedPt.y > distanceVecMax_1.y) {
            shiftedPt.y -= distanceVec.y;
        }

        if (shiftedPt.z < distanceVecMin.z) {
            shiftedPt.z += distanceVec.z;
        } else if (shiftedPt.z > distanceVecMax_1.z) {
            shiftedPt.z -= distanceVec.z;
        }
        //update shifted centroids
        xCM -= shiftedPt.x;
        yCM -= shiftedPt.y;
        zCM -= shiftedPt.z;

        //shift back centroids
        xCM += shiftVec.x * oldCell->volume;
        yCM += shiftVec.y * oldCell->volume;
        zCM += shiftVec.z * oldCell->volume;

        //Check if CM is in the allowed area
        if (xCM / (float) oldCell->volume < allowedAreaMin.x) {
            xCM += distanceVec.x * oldCell->volume;
        } else if (xCM / (float) oldCell->volume >
                   allowedAreaMax.x) { //will allow to have xCM/vol slightly bigger (by 1) value than max lattice point
            //to avoid rollovers for unsigned int from oldCell->xCM

            xCM -= distanceVec.x*oldCell->volume;

        }

        if (yCM / (float) oldCell->volume < allowedAreaMin.y) {
            yCM += distanceVec.y * oldCell->volume;
        } else if (yCM / (float) oldCell->volume > allowedAreaMax.y) {
            yCM -= distanceVec.y * oldCell->volume;
        }

        if (zCM / (float) oldCell->volume < allowedAreaMin.z) {
            zCM += distanceVec.z * oldCell->volume;
        } else if (zCM / (float) oldCell->volume > allowedAreaMax.z) {
            zCM -= distanceVec.z * oldCell->volume;
        }

        oldCell->xCM = xCM;
        oldCell->yCM = yCM;
        oldCell->zCM = zCM;

        if (oldCell->volume) {
            oldCell->xCOM = oldCell->xCM / oldCell->volume;
            oldCell->yCOM = oldCell->yCM / oldCell->volume;
            oldCell->zCOM = oldCell->zCM / oldCell->volume;
        } else {
            oldCell->xCOM = 0.0;
            oldCell->yCOM = 0.0;
            oldCell->zCOM = 0.0;
        }

		if(potts->checkIfFrozen(oldCell->type)){
			oldCell->xCOMPrev= oldCell->xCM/(oldCell->volume);
			oldCell->yCOMPrev= oldCell->yCM/(oldCell->volume);
			oldCell->zCOMPrev= oldCell->zCM/(oldCell->volume);
		}
	}

    if (newCell) {
        xo = newCell->xCM;
        yo = newCell->yCM;
        zo = newCell->zCM;

        x = newCell->xCM + pt.x;
        y = newCell->yCM + pt.y;
        z = newCell->zCM + pt.z;

        if (newCell->volume == 1) {
            shiftVec.x = 0;
            shiftVec.y = 0;
            shiftVec.z = 0;
        } else {
            shiftVec.x =
                    (newCell->xCM / (newCell->volume - 1) - ((int) fieldDimTrans.x) / 2) * boundaryConditionIndicator.x;
            shiftVec.y =
                    (newCell->yCM / (newCell->volume - 1) - ((int) fieldDimTrans.y) / 2) * boundaryConditionIndicator.y;
            shiftVec.z =
                    (newCell->zCM / (newCell->volume - 1) - ((int) fieldDimTrans.z) / 2) * boundaryConditionIndicator.z;
        }

        //shift CM to approximately center of lattice , new centroids are:
        xCM = newCell->xCM - shiftVec.x * (newCell->volume - 1);
        yCM = newCell->yCM - shiftVec.y * (newCell->volume - 1);
        zCM = newCell->zCM - shiftVec.z * (newCell->volume - 1);
        //Now shift pt
        shiftedPt = ptTrans;
        shiftedPt -= shiftVec;

        //making sure that shifted point is in the lattice
        if (shiftedPt.x < distanceVecMin.x) {
            shiftedPt.x += distanceVec.x;
        } else if (shiftedPt.x > distanceVecMax_1.x) {
            shiftedPt.x -= distanceVec.x;
		}  

        if (shiftedPt.y < distanceVecMin.y) {
            shiftedPt.y += distanceVec.y;
        } else if (shiftedPt.y > distanceVecMax_1.y) {
            shiftedPt.y -= distanceVec.y;
        }

        if (shiftedPt.z < distanceVecMin.z) {
            shiftedPt.z += distanceVec.z;
        } else if (shiftedPt.z > distanceVecMax_1.z) {
            shiftedPt.z -= distanceVec.z;
        }

        //update shifted centroids
        xCM += shiftedPt.x;
        yCM += shiftedPt.y;
        zCM += shiftedPt.z;

        //shift back centroids
        xCM += shiftVec.x * newCell->volume;
        yCM += shiftVec.y * newCell->volume;
        zCM += shiftVec.z * newCell->volume;

        //Check if CM is in the lattice
        if (xCM / (float) newCell->volume < allowedAreaMin.x) {
            xCM += distanceVec.x * newCell->volume;
        } else if (xCM / (float) newCell->volume >
                   allowedAreaMax.x) { //will allow to have xCM/vol slightly bigger (by 1) value than max lattice point
            //to avoid rollovers for unsigned int from oldCell->xCM
            xCM -= distanceVec.x * newCell->volume;
        }

        if (yCM / (float) newCell->volume < allowedAreaMin.y) {
            yCM += distanceVec.y * newCell->volume;
        } else if (yCM / (float) newCell->volume > allowedAreaMax.y) {
            yCM -= distanceVec.y * newCell->volume;
        }

        if (zCM / (float) newCell->volume < allowedAreaMin.z) {
            zCM += distanceVec.z * newCell->volume;
        } else if (zCM / (float) newCell->volume > allowedAreaMax.z) {
            zCM -= distanceVec.z * newCell->volume;
        }

        newCell->xCM = xCM;
        newCell->yCM = yCM;
        newCell->zCM = zCM;

        //storing actual center of mass
        newCell->xCOM = newCell->xCM / newCell->volume;
        newCell->yCOM = newCell->yCM / newCell->volume;
        newCell->zCOM = newCell->zCM / newCell->volume;

        if (potts->checkIfFrozen(newCell->type)) {

            newCell->xCOMPrev = newCell->xCM / (newCell->volume);
            newCell->yCOMPrev = newCell->yCM / (newCell->volume);
            newCell->zCOMPrev = newCell->zCM / (newCell->volume);

		}
	}
}

void CenterOfMassPlugin::handleEvent(CC3DEvent &_event) {
    if (_event.id != LATTICE_RESIZE) {
        return;
    }
    CC3DEventLatticeResize ev = static_cast<CC3DEventLatticeResize &>(_event);

    Dim3D shiftVec = ev.shiftVec;

    CellInventory &cellInventory = potts->getCellInventory();
    CellInventory::cellInventoryIterator cInvItr;
    CellG *cell;

    for (cInvItr = cellInventory.cellInventoryBegin(); cInvItr != cellInventory.cellInventoryEnd(); ++cInvItr) {
        cell = cInvItr->second;

        cell->xCOM+=shiftVec.x;
		cell->yCOM+=shiftVec.y;
		cell->zCOM+=shiftVec.z;

        cell->xCOMPrev += shiftVec.x;
        cell->yCOMPrev += shiftVec.y;
        cell->zCOMPrev += shiftVec.z;


        cell->xCM += shiftVec.x * cell->volume;
        cell->yCM += shiftVec.y * cell->volume;
        cell->zCM += shiftVec.z * cell->volume;


    }

}

std::string CenterOfMassPlugin::toString() { return "CenterOfMass"; }

std::string CenterOfMassPlugin::steerableName() { return toString(); }

void CenterOfMassPlugin::field3DCheck(const Point3D &pt, CellG *newCell, CellG *oldCell) {

}
