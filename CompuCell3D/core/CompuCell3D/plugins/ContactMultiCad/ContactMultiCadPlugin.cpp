#include <CompuCell3D/CC3D.h>

using namespace CompuCell3D;

#include <CompuCell3D/plugins/NeighborTracker/NeighborTrackerPlugin.h>

#include "ContactMultiCadPlugin.h"
#include <Logger/CC3DLogger.h>

ContactMultiCadPlugin::ContactMultiCadPlugin() :
        xmlData(0),
        contactEnergyPtr(&ContactMultiCadPlugin::contactEnergyLinear),
        weightDistance(false) {}

ContactMultiCadPlugin::~ContactMultiCadPlugin() {
}

void ContactMultiCadPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
    xmlData = _xmlData;
    sim = simulator;
    potts = simulator->getPotts();

    potts->getCellFactoryGroupPtr()->registerClass(&contactMultiCadDataAccessor);

    potts->registerEnergyFunctionWithName(this, "ContactMultiCad");
    simulator->registerSteerableObject(this);

}

void ContactMultiCadPlugin::extraInit(Simulator *simulator) {
    update(xmlData, true);
}


double ContactMultiCadPlugin::changeEnergy(const Point3D &pt,
    const CellG *newCell,
    const CellG *oldCell) {

    double energy = 0;
    unsigned int token = 0;
    double distance = 0;
    //   Point3D n;
    Neighbor neighbor;

    CellG *nCell = 0;
    WatchableField3D < CellG * > *fieldG = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();
    if (weightDistance) {
        for (unsigned int nIdx = 0; nIdx <= maxNeighborIndex; ++nIdx) {
            neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D &>(pt), nIdx);

            if (!neighbor.distance) {
                //if distance is 0 then the neighbor returned is invalid
                continue;
            }

            distance = neighbor.distance;

            nCell = fieldG->get(neighbor.pt);

            if (nCell != oldCell) {
                if ((nCell != 0) && (oldCell != 0)) {
                    if ((nCell->clusterId) != (oldCell->clusterId)) {
                        energy -= (this->*contactEnergyPtr)(oldCell, nCell) / neighbor.distance;
                    }
                } else {
                    energy -= (this->*contactEnergyPtr)(oldCell, nCell) / neighbor.distance;
                }

            }
            if (nCell != newCell) {
                if ((newCell != 0) && (nCell != 0)) {
                    if ((newCell->clusterId) != (nCell->clusterId)) {
                        energy += (this->*contactEnergyPtr)(newCell, nCell) / neighbor.distance;
                    }
                } else {
                    energy += (this->*contactEnergyPtr)(newCell, nCell) / neighbor.distance;

                }
            }
        }
    } else {
        //default behaviour  no energy weighting 

        for (unsigned int nIdx = 0; nIdx <= maxNeighborIndex; ++nIdx) {
            neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D &>(pt), nIdx);
            if (!neighbor.distance) {
                //if distance is 0 then the neighbor returned is invalid
                continue;
            }
            nCell = fieldG->get(neighbor.pt);

            if (nCell != oldCell) {
                if ((nCell != 0) && (oldCell != 0)) {
                    if ((nCell->clusterId) != (oldCell->clusterId)) {
                        energy -= (this->*contactEnergyPtr)(oldCell, nCell);
                    }
                } else {
                    energy -= (this->*contactEnergyPtr)(oldCell, nCell);
                }
            }
            if (nCell != newCell) {
                if ((newCell != 0) && (nCell != 0)) {
                    if ((newCell->clusterId) != (nCell->clusterId)) {
                        energy += (this->*contactEnergyPtr)(newCell, nCell);
                    }
                } else {
                    energy += (this->*contactEnergyPtr)(newCell, nCell);

                }
            }
        }

    }

    return energy;
}


double ContactMultiCadPlugin::contactEnergyLinear(const CellG *cell1, const CellG *cell2) {

    CellG *cell;
    CellG *neighbor;

    double energy = 0.0;

    if (cell1) {
        cell = const_cast<CellG *>(cell1);
        neighbor = const_cast<CellG *>(cell2);
    } else {
        cell = const_cast<CellG *>(cell2);
        neighbor = const_cast<CellG *>(cell1);
    }


    //adding "regular" contact energy
    //The minus sign is because we are adding "regular" energy to the energy expression
    energy = energyOffset + contactEnergy(cell,
                                          neighbor);
    //thus when using energyOffset-energy expression we need to compensate for extra minus sign
    if (neighbor) {

        vector<float> &jVecCell = contactMultiCadDataAccessor.get(cell->extraAttribPtr)->jVec;
        vector<float> &jVecNeighbor = contactMultiCadDataAccessor.get(neighbor->extraAttribPtr)->jVec;
        for (int i = 0; i < numberOfCadherins; ++i)
            for (int j = 0; j < numberOfCadherins; ++j) {


                energy -= jVecCell[i] * jVecNeighbor[j] * cadherinSpecificityArray[i][j];

            }

        return energy;

    }
    else {
        return energy;

    }

}


double ContactMultiCadPlugin::contactEnergy(const CellG *cell1, const CellG *cell2) {
    return contactEnergyArray[cell1 ? cell1->type : 0][cell2 ? cell2->type : 0];
}

void ContactMultiCadPlugin::setContactEnergy(const string typeName1,
                                             const string typeName2,
                                             const double energy) {

    unsigned char type1 = automaton->getTypeId(typeName1);
    unsigned char type2 = automaton->getTypeId(typeName2);

    contactEnergyArray[type1][type2] = energy;
    contactEnergyArray[type2][type1] = energy;
}


void ContactMultiCadPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {

    automaton = potts->getAutomaton();
    if (!automaton)
        throw CC3DException(
                "CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET");
    set<unsigned char> cellTypesSet;
    contactEnergyArray.clear();

    CC3DXMLElementList energyVec = _xmlData->getElements("Energy");

    for (int i = 0; i < energyVec.size(); ++i) {

        setContactEnergy(energyVec[i]->getAttribute("Type1"), energyVec[i]->getAttribute("Type2"),
                         energyVec[i]->getDouble());

        //inserting all the types to the set (duplicate are automatically eliminated) to figure out max value of type Id
        cellTypesSet.insert(automaton->getTypeId(energyVec[i]->getAttribute("Type1")));
        cellTypesSet.insert(automaton->getTypeId(energyVec[i]->getAttribute("Type2")));

    }

    CC3D_Log(LOG_DEBUG) << "size=" << contactEnergyArray.size();
    for (auto &i: cellTypesSet)
        for (auto &j: cellTypesSet) {

            CC3D_Log(LOG_DEBUG) << "contact[" << to_string(i) << "][" << to_string(j) << "]=" << contactEnergyArray[i][j];

        }


    //Here I initialize max neighbor index for direct access to the list of neighbors
    boundaryStrategy = BoundaryStrategy::getInstance();
    maxNeighborIndex = 0;

    if (_xmlData->getFirstElement("WeightEnergyByDistance")) {
        weightDistance = true;
    }


    if (_xmlData->getFirstElement("Depth")) {
        maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromDepth(
                _xmlData->getFirstElement("Depth")->getDouble());
    } else {
        if (_xmlData->getFirstElement("NeighborOrder")) {

            maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(
                    _xmlData->getFirstElement("NeighborOrder")->getUInt());
        } else {
            maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);

        }

    }
    if (_xmlData->findElement("EnergyOffset")) {
        energyOffset = _xmlData->getFirstElement("EnergyOffset")->getDouble();
    }
    if (_xmlData->findElement("ContactFunctionType")) {
        contactFunctionType = _xmlData->getFirstElement("ContactFunctionType")->getText();
        changeToLower(contactFunctionType);

        if (contactFunctionType == "linear") {
            contactEnergyPtr = &ContactMultiCadPlugin::contactEnergyLinear;
        }

    }
    CC3D_Log(LOG_DEBUG) << "Contact maxNeighborIndex=" << maxNeighborIndex;


    unsigned int cadIndex = 0;
    cadherinNameSet.clear();
    mapCadNameToIndex.clear();
    cadherinNameOrderedVector.clear();
    vector <ContactMultiCadSpecificityCadherin> vecMultCadSpecCad;

    //Check if there is PSecificity Cadherin element present

    bool specCadFlag = _xmlData->findElement("SpecificityCadherin");

    //will store xml data in vecMultCadSpecCad for SpecificityCadherin sections
    if (specCadFlag) {
        CC3DXMLElementList specCadVec = _xmlData->getElements("SpecificityCadherin");
        for (int i = 0; i < specCadVec.size(); ++i) {
            ContactMultiCadSpecificityCadherin cmcsc;
            CC3D_Log(LOG_DEBUG) << "BEFORE GETTING LIST OF SPECIFICITY";
            CC3DXMLElementList specVec = specCadVec[i]->getElements("Specificity");
            CC3D_Log(LOG_DEBUG) << "specVec.size()=" << specVec.size();

            for (int j = 0; j < specVec.size(); ++j) {
                cmcsc.Specificity(specVec[j]->getAttribute("Cadherin1"), specVec[j]->getAttribute("Cadherin2"),
                                  specVec[j]->getDouble());
                CC3D_Log(LOG_DEBUG) << "cmcsc.cadherinNameLocalSet.size()=" << cmcsc.cadherinNameLocalSet.size();
                CC3D_Log(LOG_DEBUG) << "Cadherin1=" << specVec[j]->getAttribute("Cadherin1") << " Cadherin2"
                     << specVec[j]->getAttribute("Cadherin1") << " spec=" << specVec[j]->getDouble();
            }

            vecMultCadSpecCad.push_back(cmcsc);

        }

        //copy all set elements to a master set - cadherinNameSet, defined in ContactMultiCadEnergy class
        for (int i = 0; i < vecMultCadSpecCad.size(); ++i) {
            std::set <std::string> &cadherinNameLocalSetRef = vecMultCadSpecCad[i].cadherinNameLocalSet;
            CC3D_Log(LOG_DEBUG) << "cadherinNameLocalSetRef.size()=" << cadherinNameLocalSetRef.size();

            cadherinNameSet.insert(cadherinNameLocalSetRef.begin(), cadherinNameLocalSetRef.end());
        }


        for (set<string>::iterator sitr = cadherinNameSet.begin(); sitr != cadherinNameSet.end(); ++sitr) {

            mapCadNameToIndex.insert(make_pair(*sitr, cadIndex));
            cadherinNameOrderedVector.push_back(*sitr);
            ++cadIndex;
        }


        numberOfCadherins = cadherinNameOrderedVector.size();
        CC3D_Log(LOG_DEBUG) << "numberOfCadherins=" << numberOfCadherins;
        //allocate and initialize cadherinSpecificityArray 

        cadherinSpecificityArray.assign(numberOfCadherins, vector<double>(numberOfCadherins, 0.));

        map<string, unsigned int>::iterator mitr_i;
        map<string, unsigned int>::iterator mitr_j;

        cadherinDataList.clear();

        for (int i = 0; i < vecMultCadSpecCad.size(); ++i) {
            std::vector <CadherinData> &cadherinDataVecRef = vecMultCadSpecCad[i].specificityCadherinTuppleVec;

            cadherinDataList.insert(cadherinDataList.end(), cadherinDataVecRef.begin(), cadherinDataVecRef.end());

        }

        for (list<CadherinData>::iterator litr = cadherinDataList.begin(); litr != cadherinDataList.end(); ++litr) {
            mitr_i = mapCadNameToIndex.find(litr->cad1Name);
            mitr_j = mapCadNameToIndex.find(litr->cad2Name);

            int i = mitr_i->second;
            int j = mitr_j->second;
            cadherinSpecificityArray[i][j] = litr->specificity;
            cadherinSpecificityArray[j][i] = cadherinSpecificityArray[i][j];

        }

        for (int i = 0; i < numberOfCadherins; ++i)
            for (int j = 0; j < numberOfCadherins; ++j) {
                CC3D_Log(LOG_DEBUG) <<  "specificity[" << i << "][" << j << "]=" << cadherinSpecificityArray[i][j];
            }

    }
    CC3D_Log(LOG_DEBUG) << "GOT HERE INSIDE UPDATE";


}

std::string ContactMultiCadPlugin::toString() {
    return "ContactMultiCad";
}


std::string ContactMultiCadPlugin::steerableName() {
    return toString();
}


