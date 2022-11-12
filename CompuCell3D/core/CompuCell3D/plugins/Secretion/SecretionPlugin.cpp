
#include <CompuCell3D/CC3D.h>

using namespace CompuCell3D;

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/plugins/PixelTracker/PixelTrackerPlugin.h>
#include <CompuCell3D/plugins/BoundaryPixelTracker/BoundaryPixelTrackerPlugin.h>
#include <CompuCell3D/steppables/BoxWatcher/BoxWatcher.h>

using namespace std;

#include "SecretionPlugin.h"
#include "FieldSecretor.h"

SecretionPlugin::SecretionPlugin() :
        sim(0),
        potts(0),
        xmlData(0),
        cellFieldG(0),
        automaton(0),
        boxWatcherSteppable(0),
        pUtils(0),
        boundaryStrategy(0),
        maxNeighborIndex(0),
        pixelTrackerPlugin(0),
        boundaryPixelTrackerPlugin(0),
        disablePixelTracker(false),
        disableBoundaryPixelTracker(false) {}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
SecretionPlugin::~SecretionPlugin() {}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void SecretionPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

    // IMPORTANT: listing secretion data inside secretion plugin will cause big slowdown of the program. The slowdown is even worse with multiple processors. This has to do
    // with necessity to synchronize between threads when executing secretion i.e. we have to make sure that only one thread executes secretion and
    // because we do the checks every pixel copy attempt we get really bad performance with multiple cores.
    // The bottom line: this plugin shuld be only used to implement custom secretion i.e. on a per-cell basis. Secretion by type should be specified in the PDE solver
    // REMARK: Secretion plugin is to be used to do "per cell" secretion (using python scripting)

    xmlData = _xmlData;
    sim = simulator;

    potts = sim->getPotts();
    cellFieldG = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();

    fieldDim = cellFieldG->getDim();

    pUtils = simulator->getParallelUtils();

    automaton = potts->getAutomaton();

    potts->registerFixedStepper(this,
                                true); //by putting true flag as second argument I ensure that secretion fixed stepper will be registered as first module
    //This is quick hack though not a very robust solution. We have to write CC3D scheduler...

    sim->registerSteerableObject(this);

    boundaryStrategy = BoundaryStrategy::getInstance();
    maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);


    bool comPluginAlreadyRegisteredFlag;
    Plugin *plugin = Simulator::pluginManager.get("CenterOfMass",
                                                  &comPluginAlreadyRegisteredFlag); //this will load Center Of Mass plugin if not already loaded
    if (!comPluginAlreadyRegisteredFlag)
        plugin->init(simulator);


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void SecretionPlugin::extraInit(Simulator *simulator) {

    update(xmlData, true);

    bool useBoxWatcher = false;
    for (int i = 0; i < secretionDataPVec.size(); ++i) {
        if (secretionDataPVec[i].useBoxWatcher) {
            useBoxWatcher = true;
            break;
        }
    }
    bool steppableAlreadyRegisteredFlag;
    if (useBoxWatcher) {
        boxWatcherSteppable = (BoxWatcher *) Simulator::steppableManager.get("BoxWatcher",
                                                                             &steppableAlreadyRegisteredFlag);
        if (!steppableAlreadyRegisteredFlag)
            boxWatcherSteppable->init(simulator);
    }

    bool pixelTrackerAlreadyRegisteredFlag;
    if (!disablePixelTracker) {
        pixelTrackerPlugin = (PixelTrackerPlugin *) Simulator::pluginManager.get("PixelTracker",
                                                                                 &pixelTrackerAlreadyRegisteredFlag);
        if (!pixelTrackerAlreadyRegisteredFlag) {
            pixelTrackerPlugin->init(simulator);
        }
    }

    bool boundaryPixelTrackerAlreadyRegisteredFlag;
    if (!disableBoundaryPixelTracker) {
        boundaryPixelTrackerPlugin = (BoundaryPixelTrackerPlugin *) Simulator::pluginManager.get("BoundaryPixelTracker",
                                                                                                 &boundaryPixelTrackerAlreadyRegisteredFlag);
        if (!boundaryPixelTrackerAlreadyRegisteredFlag) {
            boundaryPixelTrackerPlugin->init(simulator);
        }
    }


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Field3D<float> *SecretionPlugin::getConcentrationFieldByName(std::string _fieldName) {
    std::map < std::string, Field3D < float > * > &fieldMap = sim->getConcentrationFieldNameMap();
    std::map < std::string, Field3D < float > * > ::iterator
    mitr;
    mitr = fieldMap.find(_fieldName);
    if (mitr != fieldMap.end()) {
        return mitr->second;
    } else {
        return 0;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
FieldSecretor SecretionPlugin::getFieldSecretor(std::string _fieldName) {

    FieldSecretor fieldSecretor;

    fieldSecretor.concentrationFieldPtr = getConcentrationFieldByName(_fieldName);
    fieldSecretor.pixelTrackerPlugin = pixelTrackerPlugin;
    fieldSecretor.boundaryPixelTrackerPlugin = boundaryPixelTrackerPlugin;
    fieldSecretor.boundaryStrategy = boundaryStrategy;
    fieldSecretor.maxNeighborIndex = maxNeighborIndex;
    fieldSecretor.cellFieldG = cellFieldG;

    return fieldSecretor;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void SecretionPlugin::step(){
	CC3D_Log(LOG_TRACE) << "inside STEP SECRETION PLUGIN";
    unsigned int currentStep;
    unsigned int currentAttempt;
    unsigned int numberOfAttempts;


    currentStep = sim->getStep();
    currentAttempt = potts->getCurrentAttempt();
    numberOfAttempts = potts->getNumberOfAttempts();


    for (unsigned int i = 0; i < secretionDataPVec.size(); ++i) {
        int reminder = (numberOfAttempts % (secretionDataPVec[i].timesPerMCS + 1));
        int ratio = (numberOfAttempts / (secretionDataPVec[i].timesPerMCS + 1));
        if (!((currentAttempt - reminder) % ratio) && currentAttempt > reminder) {
            for (unsigned int j = 0; j < secretionDataPVec[i].secretionFcnPtrVec.size(); ++j) {
                (this->*secretionDataPVec[i].secretionFcnPtrVec[j])(i);

            }


        }
    }

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void SecretionPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {

    secretionDataPVec.clear();
    if (_xmlData->findElement("DisablePixelTracker")) {
        disablePixelTracker = true;
    }
    if (_xmlData->findElement("DisableBoundaryPixelTracker")) {
        disableBoundaryPixelTracker = true;
    }

    CC3DXMLElementList secrXMLVec = _xmlData->getElements("Field");

    // we unregister secretion fixed stepper when there is no XML secretion data.
    // IMPORTANT: listing secretion data inside secretion plugin will cause big slowdown of the program. The slowdown is even worse with multiple processors. This has to do
    // with necessity to synchronize between threads when executing secretion i.e. we have to make sure that only one thread executes secretion and
    // because we do the checks every pixel copy attempt we get really bad performance with multiple cores.
    // The bottom line: this plugin should be only used to implement custom secretion i.e. on a per-cell basis. Secretion by type should be specified in the PDE solver
    // REMARK: Secretion plugin is to be used to do "per cell" secretion (using python scripting)

    if (!secrXMLVec.size()) {
        potts->unregisterFixedStepper(this);
    }

    for (unsigned int i = 0; i < secrXMLVec.size(); ++i) {

        secretionDataPVec.push_back(SecretionDataP());

        SecretionDataP &secrData = secretionDataPVec[secretionDataPVec.size() - 1];
        secrData.update(secrXMLVec[i]);
        secrData.setAutomaton(potts->getAutomaton());

    }

    for (unsigned int i = 0; i < secretionDataPVec.size(); ++i) {
        secretionDataPVec[i].initialize(potts->getAutomaton());
    }

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void SecretionPlugin::secreteSingleField(unsigned int idx) {

    SecretionDataP &secrData = secretionDataPVec[idx];

    float maxUptakeInMedium = 0.0;
    float relativeUptakeRateInMedium = 0.0;
    float secrConstMedium = 0.0;

    std::map<unsigned char, float>::iterator mitrShared;
    std::map<unsigned char, float>::iterator end_mitr = secrData.typeIdSecrConstMap.end();
    std::map<unsigned char, UptakeDataP>::iterator mitrUptakeShared;
    std::map<unsigned char, UptakeDataP>::iterator end_mitrUptake = secrData.typeIdUptakeDataMap.end();


    Field3D<float> &concentrationField = *getConcentrationFieldByName(secrData.fieldName);


    bool doUptakeFlag = false;
    bool uptakeInMediumFlag = false;
    bool secreteInMedium = false;
    //the assumption is that medium has type ID 0
    mitrShared = secrData.typeIdSecrConstMap.find(automaton->getTypeId("Medium"));

    if (mitrShared != end_mitr) {
        secreteInMedium = true;
        secrConstMedium = mitrShared->second;
    }

    //uptake for medium setup
    if (secrData.typeIdUptakeDataMap.size()) {
        doUptakeFlag = true;
    }
    //uptake for medium setup
    if (doUptakeFlag) {
        mitrUptakeShared = secrData.typeIdUptakeDataMap.find(automaton->getTypeId("Medium"));
        if (mitrUptakeShared != end_mitrUptake) {
            maxUptakeInMedium = mitrUptakeShared->second.maxUptake;
            relativeUptakeRateInMedium = mitrUptakeShared->second.relativeUptakeRate;
            uptakeInMediumFlag = true;

        }
    }



    //	//HAVE TO WATCH OUT FOR SHARED/PRIVATE VARIABLES

    if (secrData.useBoxWatcher) {

        unsigned x_min = 1, x_max = fieldDim.x + 1;
        unsigned y_min = 1, y_max = fieldDim.y + 1;
        unsigned z_min = 1, z_max = fieldDim.z + 1;

        Dim3D minDimBW;
        Dim3D maxDimBW;
        Point3D minCoordinates = *(boxWatcherSteppable->getMinCoordinatesPtr());
        Point3D maxCoordinates = *(boxWatcherSteppable->getMaxCoordinatesPtr());
        CC3D_Log(LOG_TRACE) << "FLEXIBLE DIFF SOLVER maxCoordinates="<<maxCoordinates<<" minCoordinates="<<minCoordinates;
        x_min = minCoordinates.x + 1;
        x_max = maxCoordinates.x + 1;
        y_min = minCoordinates.y + 1;
        y_max = maxCoordinates.y + 1;
        z_min = minCoordinates.z + 1;
        z_max = maxCoordinates.z + 1;

        minDimBW = Dim3D(x_min, y_min, z_min);
        maxDimBW = Dim3D(x_max, y_max, z_max);
        pUtils->calculateFESolverPartitionWithBoxWatcher(minDimBW, maxDimBW);


    }


    pUtils->prepareParallelRegionFESolvers(secrData.useBoxWatcher);


#pragma omp parallel
    {

        CellG *currentCellPtr;

        float currentConcentration;
        float secrConst;


        std::map<unsigned char, float>::iterator mitr;
        std::map<unsigned char, UptakeDataP>::iterator mitrUptake;

        Point3D pt;
        int threadNumber = pUtils->getCurrentWorkNodeNumber();

        Dim3D minDim;
        Dim3D maxDim;

        if (secrData.useBoxWatcher) {
            minDim = pUtils->getFESolverPartitionWithBoxWatcher(threadNumber).first;
            maxDim = pUtils->getFESolverPartitionWithBoxWatcher(threadNumber).second;

        } else {
            minDim = pUtils->getFESolverPartition(threadNumber).first;
            maxDim = pUtils->getFESolverPartition(threadNumber).second;
        }
        //correcting for (1,1,1) shift in concentration fields used by solvers
        minDim -= Dim3D(1, 1, 1);
        maxDim -= Dim3D(1, 1, 1);

        for (int z = minDim.z; z < maxDim.z; z++)
            for (int y = minDim.y; y < maxDim.y; y++)
                for (int x = minDim.x; x < maxDim.x; x++) {

                    pt = Point3D(x, y, z);
                    currentCellPtr = cellFieldG->getQuick(pt);

                    currentConcentration = concentrationField.get(pt);

                    if (secreteInMedium && !currentCellPtr) {
                        concentrationField.set(pt, currentConcentration + secrConstMedium);
                    }

                    if (currentCellPtr) {
                        mitr = secrData.typeIdSecrConstMap.find(currentCellPtr->type);
                        if (mitr != end_mitr) {
                            secrConst = mitr->second;
                            concentrationField.set(pt, currentConcentration + secrConst);


                        }
                    }

                    if (doUptakeFlag) {

                        if (uptakeInMediumFlag && !currentCellPtr) {
                            if (currentConcentration * relativeUptakeRateInMedium > maxUptakeInMedium) {
                                concentrationField.set(pt, concentrationField.get(pt) - maxUptakeInMedium);
                            } else {
                                concentrationField.set(pt, concentrationField.get(pt) -
                                                           currentConcentration * relativeUptakeRateInMedium);
                            }
                        }
                        if (currentCellPtr) {

                            mitrUptake = secrData.typeIdUptakeDataMap.find(currentCellPtr->type);
                            if (mitrUptake != end_mitrUptake) {
                                if (currentConcentration * mitrUptake->second.relativeUptakeRate >
                                    mitrUptake->second.maxUptake) {
                                    concentrationField.set(pt,
                                                           concentrationField.get(pt) - mitrUptake->second.maxUptake);

                                } else {
                                    concentrationField.set(pt, concentrationField.get(pt) - currentConcentration *
                                                                                            mitrUptake->second.relativeUptakeRate);

                                }
                            }
                        }
                    }
                }
    }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void SecretionPlugin::secreteOnContactSingleField(unsigned int idx) {

    SecretionDataP &secrData = secretionDataPVec[idx];

    std::map<unsigned char, SecretionOnContactDataP>::iterator mitrShared;
    std::map<unsigned char, SecretionOnContactDataP>::iterator end_mitr = secrData.typeIdSecrOnContactDataMap.end();


    Field3D<float> &concentrationField = *getConcentrationFieldByName(secrData.fieldName);

    std::map<unsigned char, float> *contactCellMapMediumPtr;
    std::map<unsigned char, float> *contactCellMapPtr;


    bool secreteInMedium = false;
    //the assumption is that medium has type ID 0
    mitrShared = secrData.typeIdSecrOnContactDataMap.find(automaton->getTypeId("Medium"));

    if (mitrShared != end_mitr) {
        secreteInMedium = true;
        contactCellMapMediumPtr = &(mitrShared->second.contactCellMap);
    }


    //HAVE TO WATCH OUT FOR SHARED/PRIVATE VARIABLES


    if (secrData.useBoxWatcher) {

        unsigned x_min = 1, x_max = fieldDim.x + 1;
        unsigned y_min = 1, y_max = fieldDim.y + 1;
        unsigned z_min = 1, z_max = fieldDim.z + 1;

        Dim3D minDimBW;
        Dim3D maxDimBW;
        Point3D minCoordinates = *(boxWatcherSteppable->getMinCoordinatesPtr());
        Point3D maxCoordinates = *(boxWatcherSteppable->getMaxCoordinatesPtr());

        x_min = minCoordinates.x + 1;
        x_max = maxCoordinates.x + 1;
        y_min = minCoordinates.y + 1;
        y_max = maxCoordinates.y + 1;
        z_min = minCoordinates.z + 1;
        z_max = maxCoordinates.z + 1;

        minDimBW = Dim3D(x_min, y_min, z_min);
        maxDimBW = Dim3D(x_max, y_max, z_max);
        pUtils->calculateFESolverPartitionWithBoxWatcher(minDimBW, maxDimBW);

    }


    pUtils->prepareParallelRegionFESolvers(secrData.useBoxWatcher);
#pragma omp parallel
    {

        std::map<unsigned char, SecretionOnContactDataP>::iterator mitr;
        std::map<unsigned char, float>::iterator mitrTypeConst;

        float currentConcentration;
        float secrConst;
        float secrConstMedium = 0.0;

        CellG *currentCellPtr;
        Point3D pt;
        Neighbor n;
        CellG *nCell = 0;
        WatchableField3D < CellG * > *fieldG = (WatchableField3D < CellG * > *)
        potts->getCellFieldG();
        unsigned char type;

        int threadNumber = pUtils->getCurrentWorkNodeNumber();

        Dim3D minDim;
        Dim3D maxDim;

        if (secrData.useBoxWatcher) {
            minDim = pUtils->getFESolverPartitionWithBoxWatcher(threadNumber).first;
            maxDim = pUtils->getFESolverPartitionWithBoxWatcher(threadNumber).second;

        } else {
            minDim = pUtils->getFESolverPartition(threadNumber).first;
            maxDim = pUtils->getFESolverPartition(threadNumber).second;
        }
        //correcting for (1,1,1) shift in concetration fields used by solvers
        minDim -= Dim3D(1, 1, 1);
        maxDim -= Dim3D(1, 1, 1);


        for (int z = minDim.z; z < maxDim.z; z++)
            for (int y = minDim.y; y < maxDim.y; y++)
                for (int x = minDim.x; x < maxDim.x; x++) {
                    pt = Point3D(x, y, z);
                    ///**
                    currentCellPtr = cellFieldG->getQuick(pt);
                    //             currentCellPtr=cellFieldG->get(pt);
                    currentConcentration = concentrationField.get(pt);

                    if (secreteInMedium && !currentCellPtr) {
                        for (int i = 0; i <= maxNeighborIndex/*offsetVec.size()*/ ; ++i) {
                            n = boundaryStrategy->getNeighborDirect(pt, i);
                            if (!n.distance)//not a valid neighbor
                                continue;

                            nCell = fieldG->get(n.pt);

                            if (nCell)
                                type = nCell->type;
                            else
                                type = 0;

                            mitrTypeConst = contactCellMapMediumPtr->find(type);

                            if (mitrTypeConst != contactCellMapMediumPtr->end()) {//OK to secrete, contact detected
                                secrConstMedium = mitrTypeConst->second;

                                concentrationField.set(pt, currentConcentration + secrConstMedium);
                            }
                        }
                        continue;
                    }

                    if (currentCellPtr) {
                        mitr = secrData.typeIdSecrOnContactDataMap.find(currentCellPtr->type);
                        if (mitr != end_mitr) {

                            contactCellMapPtr = &(mitr->second.contactCellMap);

                            for (int i = 0; i <= maxNeighborIndex/*offsetVec.size() */; ++i) {

                                n = boundaryStrategy->getNeighborDirect(pt, i);
                                if (!n.distance)//not a valid neighbor
                                    continue;

                                nCell = fieldG->get(n.pt);

                                if (nCell)
                                    type = nCell->type;
                                else
                                    type = 0;

                                if (currentCellPtr == nCell)
                                    continue; //skip secretion in pixels belonging to the same cell

                                mitrTypeConst = contactCellMapPtr->find(type);
                                if (mitrTypeConst != contactCellMapPtr->end()) {//OK to secrete, contact detected
                                    secrConst = mitrTypeConst->second;

                                    concentrationField.set(pt, currentConcentration + secrConst);
                                }
                            }
                        }
                    }
                }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void SecretionPlugin::secreteConstantConcentrationSingleField(unsigned int idx) {

    SecretionDataP &secrData = secretionDataPVec[idx];

    std::map<unsigned char, float>::iterator mitrShared;
    std::map<unsigned char, float>::iterator end_mitr = secrData.typeIdSecrConstConstantConcentrationMap.end();


    float secrConstMedium = 0.0;

    Field3D<float> &concentrationField = *getConcentrationFieldByName(secrData.fieldName);


    bool secreteInMedium = false;
    //the assumption is that medium has type ID 0
    mitrShared = secrData.typeIdSecrConstConstantConcentrationMap.find(automaton->getTypeId("Medium"));

    if (mitrShared != end_mitr) {
        secreteInMedium = true;
        secrConstMedium = mitrShared->second;
    }


    //HAVE TO WATCH OUT FOR SHARED/PRIVATE VARIABLES

    if (secrData.useBoxWatcher) {

        unsigned x_min = 1, x_max = fieldDim.x + 1;
        unsigned y_min = 1, y_max = fieldDim.y + 1;
        unsigned z_min = 1, z_max = fieldDim.z + 1;

        Dim3D minDimBW;
        Dim3D maxDimBW;
        Point3D minCoordinates = *(boxWatcherSteppable->getMinCoordinatesPtr());
        Point3D maxCoordinates = *(boxWatcherSteppable->getMaxCoordinatesPtr());

        x_min = minCoordinates.x + 1;
        x_max = maxCoordinates.x + 1;
        y_min = minCoordinates.y + 1;
        y_max = maxCoordinates.y + 1;
        z_min = minCoordinates.z + 1;
        z_max = maxCoordinates.z + 1;

        minDimBW = Dim3D(x_min, y_min, z_min);
        maxDimBW = Dim3D(x_max, y_max, z_max);
        pUtils->calculateFESolverPartitionWithBoxWatcher(minDimBW, maxDimBW);

    }

    pUtils->prepareParallelRegionFESolvers(secrData.useBoxWatcher);

#pragma omp parallel
    {

        CellG *currentCellPtr;

        float currentConcentration;
        float secrConst;

        std::map<unsigned char, float>::iterator mitr;

        Point3D pt;
        int threadNumber = pUtils->getCurrentWorkNodeNumber();

        Dim3D minDim;
        Dim3D maxDim;

        if (secrData.useBoxWatcher) {
            minDim = pUtils->getFESolverPartitionWithBoxWatcher(threadNumber).first;
            maxDim = pUtils->getFESolverPartitionWithBoxWatcher(threadNumber).second;

        } else {
            minDim = pUtils->getFESolverPartition(threadNumber).first;
            maxDim = pUtils->getFESolverPartition(threadNumber).second;
        }

        //correcting for (1,1,1) shift in concetration fields used by solvers
        minDim -= Dim3D(1, 1, 1);
        maxDim -= Dim3D(1, 1, 1);

        for (int z = minDim.z; z < maxDim.z; z++)
            for (int y = minDim.y; y < maxDim.y; y++)
                for (int x = minDim.x; x < maxDim.x; x++) {

                    pt = Point3D(x, y, z);
                    currentCellPtr = cellFieldG->getQuick(pt);

                    if (secreteInMedium && !currentCellPtr) {
                        concentrationField.set(pt, secrConstMedium);
                    }

                    if (currentCellPtr) {
                        mitr = secrData.typeIdSecrConstConstantConcentrationMap.find(currentCellPtr->type);
                        if (mitr != end_mitr) {
                            secrConst = mitr->second;
                            concentrationField.set(pt, secrConst);
                        }
                    }
                }
    }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


std::string SecretionPlugin::toString() {

    return "Secretion";

}

std::string SecretionPlugin::steerableName() {

    return toString();

}
