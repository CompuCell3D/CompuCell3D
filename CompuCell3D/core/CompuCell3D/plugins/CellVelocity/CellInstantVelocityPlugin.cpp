#include "CellInstantVelocityPlugin.h"
#include <PublicUtilities/NumericalUtils.h>
#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/Potts3D/Potts3D.h>
#include <CompuCell3D/plugins/CenterOfMass/CenterOfMassPlugin.h>
#include <Utils/cldeque.h>
#include <CompuCell3D/plugins/CellVelocity/CellVelocityData.h>
#include <XMLCereal/XMLPullParser.h>
#include <XMLCereal/XMLSerializer.h>

#include "CellVelocityDataAccessor.h"
#include <iostream>

using namespace std;

namespace CompuCell3D {
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    CellInstantVelocityPlugin::CellInstantVelocityPlugin()
            : CellVelocityPlugin(), CellGChangeWatcher() {
    }

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    CellInstantVelocityPlugin::~CellInstantVelocityPlugin() {

    }

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void CellInstantVelocityPlugin::init(Simulator *_simulator) {
        CellVelocityPlugin::init(_simulator);

        potts = _simulator->getPotts();
        fieldDim = potts->getCellFieldG()->getDim();
        potts->getBoundaryXName() == "Periodic" ? boundaryConditionIndicator.x = 1 : boundaryConditionIndicator.x = 0;
        potts->getBoundaryYName() == "Periodic" ? boundaryConditionIndicator.y = 1 : boundaryConditionIndicator.y = 0;
        potts->getBoundaryZName() == "Periodic" ? boundaryConditionIndicator.z = 1 : boundaryConditionIndicator.z = 0;


    }

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void CellInstantVelocityPlugin::extraInit(Simulator *_simulator) {

        CellVelocityPlugin::extraInit(_simulator);

        potts->registerCellGChangeWatcher(this);


    }


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void CellInstantVelocityPlugin::field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell) {

        calculateInstantVelocityData(pt, newCell, oldCell);

        Coordinates3D<float> prevV;

        if (oldCell) {
            cellVelocityDataAccessorPtr->get(oldCell->extraAttribPtr)->setInstantenousVelocity(
                    ivd.oldCellV);//updating recent velocity

            cellVelocityDataAccessorPtr->get(oldCell->extraAttribPtr)->push_front(ivd.oldCellCM);//updating recent CM

        }

        if (newCell) {

            cellVelocityDataAccessorPtr->get(newCell->extraAttribPtr)->setInstantenousVelocity(
                    ivd.newCellV);//updating recent velocity

            cellVelocityDataAccessorPtr->get(newCell->extraAttribPtr)->push_front(ivd.newCellCM);//updating recent CM


        }


    }


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void CellInstantVelocityPlugin::calculateInstantVelocityData(const Point3D &pt, const CellG *newCell,
                                                                 const CellG *oldCell) {
        if (!simulator)
            return;

        //may have to update CM history here

        if (simulator->getStep() < 1) //protect from updating cell velocity during initialization
            return;

        Coordinates3D<float> v;
        Coordinates3D<float> prevV;
        Coordinates3D<float> oldCM;
        Coordinates3D<float> newCM;

        ivd.zeroAll();

        if (oldCell) {

            if (oldCell->volume > 0) {

                oldCM = cellVelocityDataAccessorPtr->get(oldCell->extraAttribPtr)->getVelocityData(0);

                newCM.XRef() = (oldCell->xCM) / (float) (oldCell->volume);
                newCM.YRef() = (oldCell->yCM) / (float) (oldCell->volume);
                newCM.ZRef() = (oldCell->zCM) / (float) (oldCell->volume);

                v.XRef() = findMin(newCM.X() - oldCM.X(), boundaryConditionIndicator.x ? fieldDim.x : 0);
                v.YRef() = findMin(newCM.Y() - oldCM.Y(), boundaryConditionIndicator.y ? fieldDim.y : 0);
                v.ZRef() = findMin(newCM.Z() - oldCM.Z(), boundaryConditionIndicator.z ? fieldDim.z : 0);

                ivd.oldCellCM = newCM; //most up2date CM for oldCell
                ivd.oldCellV = v;//most up2date instant velocity for oldCell


            }

        }

        if (newCell) {


            if (newCell->volume == 1) {

                newCM.XRef() = (newCell->xCM) / (float) (newCell->volume);
                newCM.YRef() = (newCell->yCM) / (float) (newCell->volume);
                newCM.ZRef() = (newCell->zCM) / (float) (newCell->volume);

                ivd.newCellCM = newCM; //most up2date CM for newCell
                ivd.newCellV = Coordinates3D<float>(0, 0, 0);//most up2date instant velocity for newCell


            }

            if (newCell->volume > 1) {

                oldCM = cellVelocityDataAccessorPtr->get(newCell->extraAttribPtr)->getVelocityData(0);

                newCM.XRef() = (newCell->xCM) / (float) (newCell->volume);
                newCM.YRef() = (newCell->yCM) / (float) (newCell->volume);
                newCM.ZRef() = (newCell->zCM) / (float) (newCell->volume);

                v.XRef() = findMin(newCM.X() - oldCM.X(), boundaryConditionIndicator.x ? fieldDim.x : 0);
                v.YRef() = findMin(newCM.Y() - oldCM.Y(), boundaryConditionIndicator.y ? fieldDim.y : 0);
                v.ZRef() = findMin(newCM.Z() - oldCM.Z(), boundaryConditionIndicator.z ? fieldDim.z : 0);

                ivd.newCellCM = newCM; //most up2date CM for newCell
                ivd.newCellV = v;//most up2date instant velocity for newCell



            }

        }


    }


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
};

