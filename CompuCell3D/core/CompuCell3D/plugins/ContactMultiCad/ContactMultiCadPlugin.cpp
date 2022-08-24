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

#include <CompuCell3D/plugins/NeighborTracker/NeighborTrackerPlugin.h>

#include "ContactMultiCadPlugin.h"
#include<core/CompuCell3D/CC3DLogger.h>



ContactMultiCadPlugin::ContactMultiCadPlugin() :
    xmlData(0),
    contactEnergyPtr(&ContactMultiCadPlugin::contactEnergyLinear),
    weightDistance(false)
{}

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
    WatchableField3D<CellG *> *fieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();

    if (weightDistance) {
        for (unsigned int nIdx = 0; nIdx <= maxNeighborIndex; ++nIdx) {
            neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt), nIdx);

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
                }
                else {
                    energy -= (this->*contactEnergyPtr)(oldCell, nCell) / neighbor.distance;
                }

            }
            if (nCell != newCell) {
                if ((newCell != 0) && (nCell != 0)) {
                    if ((newCell->clusterId) != (nCell->clusterId)) {
                        energy += (this->*contactEnergyPtr)(newCell, nCell) / neighbor.distance;
                    }
                }
                else {
                    energy += (this->*contactEnergyPtr)(newCell, nCell) / neighbor.distance;

                }
            }
        }
    }
    else {
        //default behaviour  no energy weighting 

        for (unsigned int nIdx = 0; nIdx <= maxNeighborIndex; ++nIdx) {
            neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt), nIdx);
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
                }
                else {
                    energy -= (this->*contactEnergyPtr)(oldCell, nCell);
                }
            }
            if (nCell != newCell) {
                if ((newCell != 0) && (nCell != 0)) {
                    if ((newCell->clusterId) != (nCell->clusterId)) {
                        energy += (this->*contactEnergyPtr)(newCell, nCell);
                    }
                }
                else {
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
    }
    else {
        cell = const_cast<CellG *>(cell2);
        neighbor = const_cast<CellG *>(cell1);
    }


    //adding "regular" contact energy
    energy = energyOffset + contactEnergy(cell, neighbor); //The minus sign is because we are adding "regular" energy to the energy expresion
                                          //thus when using energyOffset-energy expresion we need to compensate for extra minus sign
    if (neighbor) {

        vector<float> & jVecCell = contactMultiCadDataAccessor.get(cell->extraAttribPtr)->jVec;
        vector<float> & jVecNeighbor = contactMultiCadDataAccessor.get(neighbor->extraAttribPtr)->jVec;
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

    char type1 = automaton->getTypeId(typeName1);
    char type2 = automaton->getTypeId(typeName2);

    int index = getIndex(type1, type2);

    contactEnergies_t::iterator it = contactEnergies.find(index);
    ASSERT_OR_THROW(string("Contact energy for ") + typeName1 + " " + typeName2 +
        " already set!", it == contactEnergies.end());

    contactEnergies[index] = energy;
}

int ContactMultiCadPlugin::getIndex(const int type1, const int type2) const {
    if (type1 < type2) return ((type1 + 1) | ((type2 + 1) << 16));
    else return ((type2 + 1) | ((type1 + 1) << 16));
}


void ContactMultiCadPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {

    automaton = potts->getAutomaton();
    ASSERT_OR_THROW("CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET", automaton)
        set<unsigned char> cellTypesSet;
    contactEnergies.clear();

    CC3DXMLElementList energyVec = _xmlData->getElements("Energy");

    for (int i = 0; i < energyVec.size(); ++i) {

        setContactEnergy(energyVec[i]->getAttribute("Type1"), energyVec[i]->getAttribute("Type2"), energyVec[i]->getDouble());

        //inserting all the types to the set (duplicate are automatically eleminated) to figure out max value of type Id
        cellTypesSet.insert(automaton->getTypeId(energyVec[i]->getAttribute("Type1")));
        cellTypesSet.insert(automaton->getTypeId(energyVec[i]->getAttribute("Type2")));

    }

    //Now that we know all the types used in the simulation we will find size of the contactEnergyArray
    vector<unsigned char> cellTypesVector(cellTypesSet.begin(), cellTypesSet.end());//coping set to the vector

    int size = *max_element(cellTypesVector.begin(), cellTypesVector.end());
    size += 1;//if max element is e.g. 5 then size has to be 6 for an array to be properly allocated


    int index;
    contactEnergyArray.clear();
    contactEnergyArray.assign(size, vector<double>(size, 0.0));

    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j) {

            index = getIndex(cellTypesVector[i], cellTypesVector[j]);

            contactEnergyArray[i][j] = contactEnergies[index];

        }
    Log(LOG_DEBUG) << "size=" << size;
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j) {
            Log(LOG_DEBUG) << "contact[" << i << "][" << j << "]=" << contactEnergyArray[i][j];

        }


    //Here I initialize max neighbor index for direct acces to the list of neighbors 
    boundaryStrategy = BoundaryStrategy::getInstance();
    maxNeighborIndex = 0;

    if (_xmlData->getFirstElement("WeightEnergyByDistance")) {
        weightDistance = true;
    }


    if (_xmlData->getFirstElement("Depth")) {
        maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromDepth(_xmlData->getFirstElement("Depth")->getDouble());
    }
    else {
        if (_xmlData->getFirstElement("NeighborOrder")) {

            maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(_xmlData->getFirstElement("NeighborOrder")->getUInt());
        }
        else {
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
    Log(LOG_DEBUG) << "Contact maxNeighborIndex=" << maxNeighborIndex;




    unsigned int cadIndex = 0;
    cadherinNameSet.clear();
    mapCadNameToIndex.clear();
    cadherinNameOrderedVector.clear();
    vector<ContactMultiCadSpecificityCadherin> vecMultCadSpecCad;

    //Check if there is PSecificity Cadherin element present

    bool specCadFlag = _xmlData->findElement("SpecificityCadherin");

    //will store xml data in vecMultCadSpecCad for SpecificityCadherin sections
    if (specCadFlag) {
        CC3DXMLElementList specCadVec = _xmlData->getElements("SpecificityCadherin");
        for (int i = 0; i < specCadVec.size(); ++i) {
            ContactMultiCadSpecificityCadherin cmcsc;
            Log(LOG_DEBUG) << "BEFORE GETTING LIST OF SPECIFICITY";
            CC3DXMLElementList specVec = specCadVec[i]->getElements("Specificity");
            Log(LOG_DEBUG) << "specVec.size()=" << specVec.size();

            for (int j = 0; j < specVec.size(); ++j) {
                cmcsc.Specificity(specVec[j]->getAttribute("Cadherin1"), specVec[j]->getAttribute("Cadherin2"), specVec[j]->getDouble());
                Log(LOG_DEBUG) << "cmcsc.cadherinNameLocalSet.size()=" << cmcsc.cadherinNameLocalSet.size();
                Log(LOG_DEBUG) << "Cadherin1=" << specVec[j]->getAttribute("Cadherin1") << " Cadherin2" << specVec[j]->getAttribute("Cadherin1") << " spec=" << specVec[j]->getDouble();
            }

            vecMultCadSpecCad.push_back(cmcsc);

        }

        //copy all set elements to a master set - cadherinNameSet, defined in ContactMultiCadEnergy class
        for (int i = 0; i < vecMultCadSpecCad.size(); ++i) {
            std::set<std::string> & cadherinNameLocalSetRef = vecMultCadSpecCad[i].cadherinNameLocalSet;
            Log(LOG_DEBUG) << "cadherinNameLocalSetRef.size()=" << cadherinNameLocalSetRef.size();

            cadherinNameSet.insert(cadherinNameLocalSetRef.begin(), cadherinNameLocalSetRef.end());
        }


        for (set<string>::iterator sitr = cadherinNameSet.begin(); sitr != cadherinNameSet.end(); ++sitr) {

            mapCadNameToIndex.insert(make_pair(*sitr, cadIndex));
            cadherinNameOrderedVector.push_back(*sitr);
            ++cadIndex;
        }


        numberOfCadherins = cadherinNameOrderedVector.size();
        Log(LOG_DEBUG) << "numberOfCadherins=" << numberOfCadherins;
        //allocate and initialize cadherinSpecificityArray 

        cadherinSpecificityArray.assign(numberOfCadherins, vector<double>(numberOfCadherins, 0.));

        map<string, unsigned int>::iterator mitr_i;
        map<string, unsigned int>::iterator mitr_j;

        cadherinDataList.clear();

        for (int i = 0; i < vecMultCadSpecCad.size(); ++i) {
            std::vector<CadherinData> & cadherinDataVecRef = vecMultCadSpecCad[i].specificityCadherinTuppleVec;

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
                Log(LOG_DEBUG) <<  "specificity[" << i << "][" << j << "]=" << cadherinSpecificityArray[i][j];
            }

    }
    Log(LOG_DEBUG) << "GOT HERE INSIDE UPDATE";


}

std::string ContactMultiCadPlugin::toString() {
    return "ContactMultiCad";
}


std::string ContactMultiCadPlugin::steerableName() {
    return toString();
}


