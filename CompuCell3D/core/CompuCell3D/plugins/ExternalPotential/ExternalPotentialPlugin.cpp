#include <CompuCell3D/CC3D.h>

using namespace CompuCell3D;
using namespace std;


#include "ExternalPotentialPlugin.h"

namespace CompuCell3D {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ExternalPotentialPlugin::ExternalPotentialPlugin() : lambdaVec(Coordinates3D<float>(0., 0., 0.)), xmlData(0),
                                                         boundaryStrategy(0) {}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ExternalPotentialPlugin::~ExternalPotentialPlugin() {
    }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void ExternalPotentialPlugin::init(Simulator *_simulator, CC3DXMLElement *_xmlData) {

        xmlData = _xmlData;
        potts = _simulator->getPotts();
        cellFieldG = (WatchableField3D < CellG * > *)
        potts->getCellFieldG();
        simulator = _simulator;

        bool pluginAlreadyRegisteredFlag;
        //this will load VolumeTracker plugin if it is not already loaded
        Plugin *plugin = Simulator::pluginManager.get("CenterOfMass",
                                                      &pluginAlreadyRegisteredFlag);
        if (!pluginAlreadyRegisteredFlag)
            plugin->init(simulator);


        potts->registerEnergyFunctionWithName(this, "ExternalPotential");

        fieldDim = potts->getCellFieldG()->getDim();


        boundaryStrategy = BoundaryStrategy::getInstance();
        adjNeighbor.initialize(fieldDim);
        adjNeighbor_ptr = &adjNeighbor;

        if (potts->getBoundaryXName() == "Periodic") {
            adjNeighbor.setPeriodicX();
            boundaryConditionIndicator.x = 1;
        }
        if (potts->getBoundaryYName() == "Periodic") {
            adjNeighbor.setPeriodicY();
            boundaryConditionIndicator.y = 1;
        }
        if (potts->getBoundaryZName() == "Periodic") {
            adjNeighbor.setPeriodicZ();
            boundaryConditionIndicator.z = 1;
        }

        _simulator->registerSteerableObject(this);


    }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void ExternalPotentialPlugin::extraInit(Simulator *_simulator) {
        // check why registering in init gives segfault

        update(xmlData, true);
    }


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    void ExternalPotentialPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {

        bool comBasedAlgorithm = false;
        if (_xmlData->findElement("Algorithm")) {

            string algorithm = _xmlData->getFirstElement("Algorithm")->getText();

            changeToLower(algorithm);

            if (algorithm == "centerofmassbased") {
                comBasedAlgorithm = true;
            } else if (algorithm == "pixelbased") {
                comBasedAlgorithm = false;
            }

        }

        if (!_xmlData->getNumberOfChildren() ||
            (_xmlData->getNumberOfChildren() == 1 && _xmlData->findElement("Algorithm"))) {
            functionType = BYCELLID;
        } else {
            if (_xmlData->findElement("ExternalPotentialParameters"))
                functionType = BYCELLTYPE;
            else if (_xmlData->findElement("Lambda"))
                functionType = GLOBAL;
            else //in case users put garbage xml use changeEnergyByCellId
                functionType = BYCELLID;
        }

        Automaton *automaton = potts->getAutomaton();


        switch (functionType) {
            case BYCELLID:
                //set fcn ptr
                if (comBasedAlgorithm) {
                    changeEnergyFcnPtr = &ExternalPotentialPlugin::changeEnergyByCellIdCOMBased;
                } else {
                    changeEnergyFcnPtr = &ExternalPotentialPlugin::changeEnergyByCellId;
                }
                break;

            case BYCELLTYPE: {
                externalPotentialParamMap.clear();

                CC3DXMLElementList energyVec = _xmlData->getElements("ExternalPotentialParameters");

                for (int i = 0; i < energyVec.size(); ++i) {
                    ExternalPotentialParam extPotentialParam;

                    extPotentialParam.lambdaVec.x = energyVec[i]->getAttributeAsDouble("x");
                    extPotentialParam.lambdaVec.y = energyVec[i]->getAttributeAsDouble("y");
                    extPotentialParam.lambdaVec.z = energyVec[i]->getAttributeAsDouble("z");

                    extPotentialParam.typeName = energyVec[i]->getAttribute("CellType");

                    participatingTypes.insert(automaton->getTypeId(extPotentialParam.typeName));

                    externalPotentialParamMap[automaton->getTypeId(extPotentialParam.typeName)] = extPotentialParam;
                }

                //set fcn ptr
                if (comBasedAlgorithm) {
                    changeEnergyFcnPtr = &ExternalPotentialPlugin::changeEnergyByCellTypeCOMBased;
                } else {
                    changeEnergyFcnPtr = &ExternalPotentialPlugin::changeEnergyByCellType;
                }

            }
                break;

            case GLOBAL:
                //using Global Volume Energy Parameters
                lambdaVec = Coordinates3D<float>(_xmlData->getFirstElement("Lambda")->getAttributeAsDouble("x"),
                                                 _xmlData->getFirstElement("Lambda")->getAttributeAsDouble("y"),
                                                 _xmlData->getFirstElement("Lambda")->getAttributeAsDouble("z")
                );
                //set fcn ptr
                //changeEnergyFcnPtr=&ExternalPotentialPlugin::changeEnergyGlobal;
                if (comBasedAlgorithm) {
                    changeEnergyFcnPtr = &ExternalPotentialPlugin::changeEnergyGlobalCOMBased;
                } else {
                    changeEnergyFcnPtr = &ExternalPotentialPlugin::changeEnergyGlobal;
                }


                break;

            default:
                //set fcn ptr
                if (comBasedAlgorithm) {
                    changeEnergyFcnPtr = &ExternalPotentialPlugin::changeEnergyByCellIdCOMBased;
                } else {
                    changeEnergyFcnPtr = &ExternalPotentialPlugin::changeEnergyByCellId;
                }
        }

    }

    double ExternalPotentialPlugin::changeEnergyGlobalCOMBased(const Point3D &pt, const CellG *newCell,
                                                               const CellG *oldCell) {

        double energy = 0.0;
        if (oldCell) {
            Coordinates3D<double> oldCOMAfterFlip = precalculateCentroid(pt, oldCell, -1, fieldDim, boundaryStrategy);

            if (oldCell->volume > 1) {
                oldCOMAfterFlip.XRef() = oldCOMAfterFlip.X() / (float) (oldCell->volume - 1);
                oldCOMAfterFlip.YRef() = oldCOMAfterFlip.Y() / (float) (oldCell->volume - 1);
                oldCOMAfterFlip.ZRef() = oldCOMAfterFlip.Z() / (float) (oldCell->volume - 1);
            } else {

                oldCOMAfterFlip = Coordinates3D<double>(oldCell->xCM / oldCell->volume, oldCell->zCM / oldCell->volume,
                                                        oldCell->zCM / oldCell->volume);

            }

            Coordinates3D<double> oldCOMBeforeFlip(oldCell->xCM / oldCell->volume, oldCell->yCM / oldCell->volume,
                                                   oldCell->zCM / oldCell->volume);
            Coordinates3D<double> distVector = distanceVectorCoordinatesInvariant(oldCOMAfterFlip, oldCOMBeforeFlip,
                                                                                  fieldDim);
            energy += distVector.X() * lambdaVec.X() + distVector.Y() * lambdaVec.Y() + distVector.Z() * lambdaVec.Z();
        }

        if (newCell) {

            Coordinates3D<double> newCOMAfterFlip = precalculateCentroid(pt, newCell, 1, fieldDim, boundaryStrategy);


            newCOMAfterFlip.XRef() = newCOMAfterFlip.X() / (float) (newCell->volume + 1);
            newCOMAfterFlip.YRef() = newCOMAfterFlip.Y() / (float) (newCell->volume + 1);
            newCOMAfterFlip.ZRef() = newCOMAfterFlip.Z() / (float) (newCell->volume + 1);


            Coordinates3D<double> newCOMBeforeFlip(newCell->xCM / newCell->volume, newCell->yCM / newCell->volume,
                                                   newCell->zCM / newCell->volume);
            Coordinates3D<double> distVector = distanceVectorCoordinatesInvariant(newCOMAfterFlip, newCOMBeforeFlip,
                                                                                  fieldDim);

            energy += distVector.X() * lambdaVec.X() + distVector.Y() * lambdaVec.Y() + distVector.Z() * lambdaVec.Z();

        }

        return energy;
    }

    double ExternalPotentialPlugin::changeEnergyByCellTypeCOMBased(const Point3D &pt, const CellG *newCell,
                                                                   const CellG *oldCell) {

        double energy = 0.0;
        if (oldCell) {
            Coordinates3D<double> oldCOMAfterFlip = precalculateCentroid(pt, oldCell, -1, fieldDim, boundaryStrategy);

            if (oldCell->volume > 1) {
                oldCOMAfterFlip.XRef() = oldCOMAfterFlip.X() / (float) (oldCell->volume - 1);
                oldCOMAfterFlip.YRef() = oldCOMAfterFlip.Y() / (float) (oldCell->volume - 1);
                oldCOMAfterFlip.ZRef() = oldCOMAfterFlip.Z() / (float) (oldCell->volume - 1);
            } else {

                oldCOMAfterFlip = Coordinates3D<double>(oldCell->xCM / oldCell->volume, oldCell->zCM / oldCell->volume,
                                                        oldCell->zCM / oldCell->volume);

            }

            Coordinates3D<double> oldCOMBeforeFlip(oldCell->xCM / oldCell->volume, oldCell->yCM / oldCell->volume,
                                                   oldCell->zCM / oldCell->volume);
            Coordinates3D<double> distVector = distanceVectorCoordinatesInvariant(oldCOMAfterFlip, oldCOMBeforeFlip,
                                                                                  fieldDim);
            energy += distVector.X() * externalPotentialParamMap[oldCell->type].lambdaVec.X()
                      + distVector.Y() * externalPotentialParamMap[oldCell->type].lambdaVec.Y()
                      + distVector.Z() * externalPotentialParamMap[oldCell->type].lambdaVec.Z();
        }

        if (newCell) {

            Coordinates3D<double> newCOMAfterFlip = precalculateCentroid(pt, newCell, 1, fieldDim, boundaryStrategy);


            newCOMAfterFlip.XRef() = newCOMAfterFlip.X() / (float) (newCell->volume + 1);
            newCOMAfterFlip.YRef() = newCOMAfterFlip.Y() / (float) (newCell->volume + 1);
            newCOMAfterFlip.ZRef() = newCOMAfterFlip.Z() / (float) (newCell->volume + 1);


            Coordinates3D<double> newCOMBeforeFlip(newCell->xCM / newCell->volume, newCell->yCM / newCell->volume,
                                                   newCell->zCM / newCell->volume);
            Coordinates3D<double> distVector = distanceVectorCoordinatesInvariant(newCOMAfterFlip, newCOMBeforeFlip,
                                                                                  fieldDim);

            energy += distVector.X() * externalPotentialParamMap[newCell->type].lambdaVec.X()
                      + distVector.Y() * externalPotentialParamMap[newCell->type].lambdaVec.Y()
                      + distVector.Z() * externalPotentialParamMap[newCell->type].lambdaVec.Z();

        }

        return energy;
    }

    double ExternalPotentialPlugin::changeEnergyByCellIdCOMBased(const Point3D &pt, const CellG *newCell,
                                                                 const CellG *oldCell) {

        double energy = 0.0;
        if (oldCell) {
            Coordinates3D<double> oldCOMAfterFlip = precalculateCentroid(pt, oldCell, -1, fieldDim, boundaryStrategy);

            if (oldCell->volume > 1) {
                oldCOMAfterFlip.XRef() = oldCOMAfterFlip.X() / (float) (oldCell->volume - 1);
                oldCOMAfterFlip.YRef() = oldCOMAfterFlip.Y() / (float) (oldCell->volume - 1);
                oldCOMAfterFlip.ZRef() = oldCOMAfterFlip.Z() / (float) (oldCell->volume - 1);
            } else {

                oldCOMAfterFlip = Coordinates3D<double>(oldCell->xCM / oldCell->volume, oldCell->zCM / oldCell->volume,
                                                        oldCell->zCM / oldCell->volume);

            }

            Coordinates3D<double> oldCOMBeforeFlip(oldCell->xCM / oldCell->volume, oldCell->yCM / oldCell->volume,
                                                   oldCell->zCM / oldCell->volume);
            Coordinates3D<double> distVector = distanceVectorCoordinatesInvariant(oldCOMAfterFlip, oldCOMBeforeFlip,
                                                                                  fieldDim);
            energy += distVector.X() * oldCell->lambdaVecX + distVector.Y() * oldCell->lambdaVecY +
                      distVector.Z() * oldCell->lambdaVecZ;
        }

        if (newCell) {

            Coordinates3D<double> newCOMAfterFlip = precalculateCentroid(pt, newCell, 1, fieldDim, boundaryStrategy);


            newCOMAfterFlip.XRef() = newCOMAfterFlip.X() / (float) (newCell->volume + 1);
            newCOMAfterFlip.YRef() = newCOMAfterFlip.Y() / (float) (newCell->volume + 1);
            newCOMAfterFlip.ZRef() = newCOMAfterFlip.Z() / (float) (newCell->volume + 1);


            Coordinates3D<double> newCOMBeforeFlip(newCell->xCM / newCell->volume, newCell->yCM / newCell->volume,
                                                   newCell->zCM / newCell->volume);
            Coordinates3D<double> distVector = distanceVectorCoordinatesInvariant(newCOMAfterFlip, newCOMBeforeFlip,
                                                                                  fieldDim);

            energy += distVector.X() * newCell->lambdaVecX + distVector.Y() * newCell->lambdaVecY +
                      distVector.Z() * newCell->lambdaVecZ;

        }

        return energy;
    }


    double ExternalPotentialPlugin::changeEnergyGlobal(const Point3D &pt, const CellG *newCell,
                                                       const CellG *oldCell) {

        double deltaEnergyOld = 0.0;
        double deltaEnergyNew = 0.0;
        CellG *neighborPtr;

        Dim3D fieldDim = cellFieldG->getDim();


        const vector <Point3D> &neighborsOffsetVec = adjNeighbor_ptr->getAdjFace2FaceNeighborOffsetVec(pt);
        unsigned int neighborSize = neighborsOffsetVec.size();

        Point3D ptAdj;


        //x, y,or z depending in which direction we want potential gradient to act
        //I add 1 to avoid zero change of energy when z is at the boundary
        short prefferedCoordinate = pt.z + 1;
        //short deltaCoordinate;
        Coordinates3D<short> deltaCoordinate(0, 0, 0);

        ///COMMENT TO deltaCoordinate calculations
        // have to do fieldDim.z-2 because the neighbor of pt with max_z can be a pt
        // with z=0 but they are spaced by delta z=1
        // thus you need to do deltaCoordinate % (fieldDim.z-2). Otherwise if you do not do fieldDim.z-2
        // then you may get the following (max_z-0) % max_z =0, whereas it should be 1 !
CC3D_Log(LOG_TRACE) << "******************CHANGE BEGIN**************************";
        int counter = 0;
        Point3D ptFlipNeighbor = potts->getFlipNeighbor();
        Point3D deltaFlip;

        deltaFlip = pt;
        deltaFlip -= ptFlipNeighbor;

        for (unsigned int i = 0; i < neighborSize; ++i) {
            ptAdj = pt;
            ptAdj += neighborsOffsetVec[i];

            if (cellFieldG->isValid(ptAdj)) {
                ++counter;
                neighborPtr = cellFieldG->get(ptAdj);

                ///process old energy
                if (/*neighborPtr &&*/ oldCell && neighborPtr != oldCell) {

                    deltaCoordinate.XRef() = (ptAdj.x - pt.x);
                    deltaCoordinate.YRef() = (ptAdj.y - pt.y);
                    deltaCoordinate.ZRef() = (ptAdj.z - pt.z);

                    if (fabs(static_cast<double>(deltaCoordinate.X())) > 1) {

                        deltaCoordinate.XRef() =
                                (deltaCoordinate.X() > 0 ? -(deltaCoordinate.X() + 1) % (fieldDim.x - 1) :
                                 -(deltaCoordinate.X() - 1) % (fieldDim.x - 1));


                    }

                    if (fabs(static_cast<double>(deltaCoordinate.Y())) > 1) {
                        deltaCoordinate.YRef() =
                                (deltaCoordinate.Y() > 0 ? -(deltaCoordinate.Y() + 1) % (fieldDim.y - 1) :
                                 -(deltaCoordinate.Y() - 1) % (fieldDim.y - 1));
                    }


                    if (fabs(static_cast<double>(deltaCoordinate.Z())) > 1) {
                        deltaCoordinate.ZRef() =
                                (deltaCoordinate.Z() > 0 ? -(deltaCoordinate.Z() + 1) % (fieldDim.z - 1) :
                                 -(deltaCoordinate.Z() - 1) % (fieldDim.z - 1));
                    }
                    deltaEnergyOld += deltaCoordinate.X() * lambdaVec.X() +
                                      deltaCoordinate.Y() * lambdaVec.Y() +
                                      deltaCoordinate.Z() * lambdaVec.Z();

                }

                ///process new energy
                if (/*neighborPtr && */newCell && neighborPtr != newCell) {

                    deltaCoordinate.XRef() = (ptAdj.x - pt.x);
                    deltaCoordinate.YRef() = (ptAdj.y - pt.y);
                    deltaCoordinate.ZRef() = (ptAdj.z - pt.z);

                    if (fabs(static_cast<double>(deltaCoordinate.X())) > 1) {
                        deltaCoordinate.XRef() =
                                (deltaCoordinate.X() > 0 ? -(deltaCoordinate.X() + 1) % (fieldDim.x - 1) :
                                 -(deltaCoordinate.X() - 1) % (fieldDim.x - 1));

                    }

                    if (fabs(static_cast<double>(deltaCoordinate.Y())) > 1) {
                        deltaCoordinate.YRef() =
                                (deltaCoordinate.Y() > 0 ? -(deltaCoordinate.Y() + 1) % (fieldDim.y - 1) :
                                 -(deltaCoordinate.Y() - 1) % (fieldDim.y - 1));
                    }


                    if (fabs(static_cast<double>(deltaCoordinate.Z())) > 1) {
                        deltaCoordinate.ZRef() =
                                (deltaCoordinate.Z() > 0 ? -(deltaCoordinate.Z() + 1) % (fieldDim.z - 1) :
                                 -(deltaCoordinate.Z() - 1) % (fieldDim.z - 1));
                    }

                    deltaEnergyNew += deltaCoordinate.X() * lambdaVec.X() +
                                      deltaCoordinate.Y() * lambdaVec.Y() +
                                      deltaCoordinate.Z() * lambdaVec.Z();

                }


            }

        }


        return deltaEnergyNew - deltaEnergyOld;

    }


    double ExternalPotentialPlugin::changeEnergyByCellType(const Point3D &pt, const CellG *newCell,
                                                           const CellG *oldCell) {


        double deltaEnergyOld = 0.0;
        double deltaEnergyNew = 0.0;
        CellG *neighborPtr;

        Dim3D fieldDim = cellFieldG->getDim();


        const vector <Point3D> &neighborsOffsetVec = adjNeighbor_ptr->getAdjFace2FaceNeighborOffsetVec(pt);
        unsigned int neighborSize = neighborsOffsetVec.size();

        Point3D ptAdj;


        //x, y,or z depending in which direction we want potential gradient to act
        //I add 1 to avoid zero change of energy when z is at the boundary
        short prefferedCoordinate = pt.z + 1;
        //short deltaCoordinate;
        Coordinates3D<short> deltaCoordinate(0, 0, 0);

        ///COMMENT TO deltaCoordinate calculations
        // have to do fieldDim.z-2 because the neighbor of pt with max_z can be
        // a pt with z=0 but they are spaced by delta z=1
        // thus you need to do deltaCoordinate % (fieldDim.z-2). Otherwise if you do not do fieldDim.z-2
        // then you may get the following (max_z-0) % max_z =0, whereas it should be 1 !

        int counter = 0;
        Point3D ptFlipNeighbor = potts->getFlipNeighbor();
        Point3D deltaFlip;

        deltaFlip = pt;
        deltaFlip -= ptFlipNeighbor;

        for (unsigned int i = 0; i < neighborSize; ++i) {
            ptAdj = pt;
            ptAdj += neighborsOffsetVec[i];

            if (cellFieldG->isValid(ptAdj)) {
                ++counter;
                neighborPtr = cellFieldG->get(ptAdj);

                ///process old energy
                if (/*neighborPtr &&*/ oldCell && neighborPtr != oldCell &&
                                       participatingTypes.find(oldCell->type) != participatingTypes.end()) {
                    deltaCoordinate.XRef() = (ptAdj.x - pt.x);
                    deltaCoordinate.YRef() = (ptAdj.y - pt.y);
                    deltaCoordinate.ZRef() = (ptAdj.z - pt.z);

                    if (fabs(static_cast<double>(deltaCoordinate.X())) > 1) {

                        deltaCoordinate.XRef() =
                                (deltaCoordinate.X() > 0 ? -(deltaCoordinate.X() + 1) % (fieldDim.x - 1) :
                                 -(deltaCoordinate.X() - 1) % (fieldDim.x - 1));


                    }

                    if (fabs(static_cast<double>(deltaCoordinate.Y())) > 1) {
                        deltaCoordinate.YRef() =
                                (deltaCoordinate.Y() > 0 ? -(deltaCoordinate.Y() + 1) % (fieldDim.y - 1) :
                                 -(deltaCoordinate.Y() - 1) % (fieldDim.y - 1));
                    }


                    if (fabs(static_cast<double>(deltaCoordinate.Z())) > 1) {
                        deltaCoordinate.ZRef() =
                                (deltaCoordinate.Z() > 0 ? -(deltaCoordinate.Z() + 1) % (fieldDim.z - 1) :
                                 -(deltaCoordinate.Z() - 1) % (fieldDim.z - 1));
                    }

                    deltaEnergyOld += deltaCoordinate.X() * externalPotentialParamMap[oldCell->type].lambdaVec.X() +
                                      deltaCoordinate.Y() * externalPotentialParamMap[oldCell->type].lambdaVec.Y() +
                                      deltaCoordinate.Z() * externalPotentialParamMap[oldCell->type].lambdaVec.Z();

                }

                ///process new energy
                if (/*neighborPtr && */newCell && neighborPtr != newCell &&
                                       participatingTypes.find(newCell->type) != participatingTypes.end()) {

                    deltaCoordinate.XRef() = (ptAdj.x - pt.x);
                    deltaCoordinate.YRef() = (ptAdj.y - pt.y);
                    deltaCoordinate.ZRef() = (ptAdj.z - pt.z);

                    if (fabs(static_cast<double>(deltaCoordinate.X())) > 1) {
                        deltaCoordinate.XRef() =
                                (deltaCoordinate.X() > 0 ? -(deltaCoordinate.X() + 1) % (fieldDim.x - 1) :
                                 -(deltaCoordinate.X() - 1) % (fieldDim.x - 1));

                    }

                    if (fabs(static_cast<double>(deltaCoordinate.Y())) > 1) {
                        deltaCoordinate.YRef() =
                                (deltaCoordinate.Y() > 0 ? -(deltaCoordinate.Y() + 1) % (fieldDim.y - 1) :
                                 -(deltaCoordinate.Y() - 1) % (fieldDim.y - 1));
                    }


                    if (fabs(static_cast<double>(deltaCoordinate.Z())) > 1) {
                        deltaCoordinate.ZRef() =
                                (deltaCoordinate.Z() > 0 ? -(deltaCoordinate.Z() + 1) % (fieldDim.z - 1) :
                                 -(deltaCoordinate.Z() - 1) % (fieldDim.z - 1));
                    }
                    deltaEnergyNew += deltaCoordinate.X() * externalPotentialParamMap[newCell->type].lambdaVec.X() +
                                      deltaCoordinate.Y() * externalPotentialParamMap[newCell->type].lambdaVec.Y() +
                                      deltaCoordinate.Z() * externalPotentialParamMap[newCell->type].lambdaVec.Z();

                }


            }

        }

        return deltaEnergyNew - deltaEnergyOld;

    }

    double ExternalPotentialPlugin::changeEnergyByCellId(const Point3D &pt, const CellG *newCell,
                                                         const CellG *oldCell) {


        double deltaEnergyOld = 0.0;
        double deltaEnergyNew = 0.0;
        CellG *neighborPtr;

        Dim3D fieldDim = cellFieldG->getDim();


        const vector <Point3D> &neighborsOffsetVec = adjNeighbor_ptr->getAdjFace2FaceNeighborOffsetVec(pt);
        unsigned int neighborSize = neighborsOffsetVec.size();

        Point3D ptAdj;


        //x, y,or z depending in which direction we want potential gradient to act
        //I add 1 to avoid zero change of energy when z is at the boundary
        short prefferedCoordinate = pt.z + 1;
        //short deltaCoordinate;
        Coordinates3D<short> deltaCoordinate(0, 0, 0);

        ///COMMENT TO deltaCoordinate calculations
        // have to do fieldDim.z-2 because the neighbor of pt with max_z can
        // be a pt with z=0 but they are spaced by delta z=1
        // thus you need to do deltaCoordinate % (fieldDim.z-2). Otherwise if you do not do fieldDim.z-2
        // then you may get the following (max_z-0) % max_z =0, whereas it should be 1 !


        int counter = 0;
        Point3D ptFlipNeighbor = potts->getFlipNeighbor();
        Point3D deltaFlip;

        deltaFlip = pt;
        deltaFlip -= ptFlipNeighbor;

        for (unsigned int i = 0; i < neighborSize; ++i) {
            ptAdj = pt;
            ptAdj += neighborsOffsetVec[i];

            if (cellFieldG->isValid(ptAdj)) {
                ++counter;
                neighborPtr = cellFieldG->get(ptAdj);

                ///process old energy
                if (/*neighborPtr &&*/ oldCell && neighborPtr != oldCell) {

                    deltaCoordinate.XRef() = (ptAdj.x - pt.x);
                    deltaCoordinate.YRef() = (ptAdj.y - pt.y);
                    deltaCoordinate.ZRef() = (ptAdj.z - pt.z);

                    if (fabs(static_cast<double>(deltaCoordinate.X())) > 1) {

                        deltaCoordinate.XRef() =
                                (deltaCoordinate.X() > 0 ? -(deltaCoordinate.X() + 1) % (fieldDim.x - 1) :
                                 -(deltaCoordinate.X() - 1) % (fieldDim.x - 1));


                    }

                    if (fabs(static_cast<double>(deltaCoordinate.Y())) > 1) {
                        deltaCoordinate.YRef() =
                                (deltaCoordinate.Y() > 0 ? -(deltaCoordinate.Y() + 1) % (fieldDim.y - 1) :
                                 -(deltaCoordinate.Y() - 1) % (fieldDim.y - 1));
                    }


                    if (fabs(static_cast<double>(deltaCoordinate.Z())) > 1) {
                        deltaCoordinate.ZRef() =
                                (deltaCoordinate.Z() > 0 ? -(deltaCoordinate.Z() + 1) % (fieldDim.z - 1) :
                                 -(deltaCoordinate.Z() - 1) % (fieldDim.z - 1));
                    }

                    deltaEnergyOld += deltaCoordinate.X() * oldCell->lambdaVecX +
                                      deltaCoordinate.Y() * oldCell->lambdaVecY +
                                      deltaCoordinate.Z() * oldCell->lambdaVecZ;

                }

                ///process new energy
                if (/*neighborPtr && */newCell && neighborPtr != newCell) {

                    deltaCoordinate.XRef() = (ptAdj.x - pt.x);
                    deltaCoordinate.YRef() = (ptAdj.y - pt.y);
                    deltaCoordinate.ZRef() = (ptAdj.z - pt.z);

                    if (fabs(static_cast<double>(deltaCoordinate.X())) > 1) {
                        deltaCoordinate.XRef() =
                                (deltaCoordinate.X() > 0 ? -(deltaCoordinate.X() + 1) % (fieldDim.x - 1) :
                                 -(deltaCoordinate.X() - 1) % (fieldDim.x - 1));

                    }

                    if (fabs(static_cast<double>(deltaCoordinate.Y())) > 1) {
                        deltaCoordinate.YRef() =
                                (deltaCoordinate.Y() > 0 ? -(deltaCoordinate.Y() + 1) % (fieldDim.y - 1) :
                                 -(deltaCoordinate.Y() - 1) % (fieldDim.y - 1));
                    }


                    if (fabs(static_cast<double>(deltaCoordinate.Z())) > 1) {
                        deltaCoordinate.ZRef() =
                                (deltaCoordinate.Z() > 0 ? -(deltaCoordinate.Z() + 1) % (fieldDim.z - 1) :
                                 -(deltaCoordinate.Z() - 1) % (fieldDim.z - 1));
                    }

                    deltaEnergyNew += deltaCoordinate.X() * newCell->lambdaVecX +
                                      deltaCoordinate.Y() * newCell->lambdaVecY +
                                      deltaCoordinate.Z() * newCell->lambdaVecZ;

                }


            }

        }


        return deltaEnergyNew - deltaEnergyOld;

    }


    double ExternalPotentialPlugin::changeEnergy(const Point3D &pt, const CellG *newCell,
                                                 const CellG *oldCell) {

        return (this->*changeEnergyFcnPtr)(pt, newCell, oldCell);

    }



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



    std::string ExternalPotentialPlugin::toString() {
        return "ExternalPotential";
    }


    std::string ExternalPotentialPlugin::steerableName() {

        return toString();
    }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



};
