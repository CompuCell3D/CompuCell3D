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

#include "ContactPlugin.h"


ContactPlugin::ContactPlugin() :xmlData(0), weightDistance(false) {
}

ContactPlugin::~ContactPlugin() {

}

void ContactPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
    potts = simulator->getPotts();
    xmlData = _xmlData;
    simulator->getPotts()->registerEnergyFunctionWithName(this, toString());
    simulator->registerSteerableObject(this);
}

void ContactPlugin::extraInit(Simulator *simulator) {
    //cerr << "contact enter extraInit" << endl;
    update(xmlData, true);
    //cerr << "contact after update" << endl;
    Automaton * cellTypePluginAutomaton = potts->getAutomaton();
    //cerr << "contact after automaton" << endl;
    return;
    if (cellTypePluginAutomaton) {
        ASSERT_OR_THROW("The size of matrix of contact energy coefficients has must equal max_cell_type_id+1. You must list interactions coefficients between all cel types",
            contactEnergyArray.size() == ((unsigned int)cellTypePluginAutomaton->getMaxTypeId() + 1));
    }

}

void ContactPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {
    automaton = potts->getAutomaton();
    //cerr << "automaton=" << automaton << endl;
    //return;
    ASSERT_OR_THROW("CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET", automaton)
        set<unsigned char> cellTypesSet;
    contactEnergies.clear();


    //////if(potts->getDisplayUnitsFlag()){
    //////	Unit contactEnergyUnit=potts->getEnergyUnit()/powerUnit(potts->getLengthUnit(),2);




    //////	CC3DXMLElement * unitsElem=_xmlData->getFirstElement("Units"); 
    //////	if (!unitsElem){ //add Units element
    //////		unitsElem=_xmlData->attachElement("Units");
    //////	}

    //////	if(unitsElem->getFirstElement("EnergyUnit")){
    //////		unitsElem->getFirstElement("EnergyUnit")->updateElementValue(contactEnergyUnit.toString());
    //////	}else{
    //////		CC3DXMLElement * energyUnitElem = unitsElem->attachElement("EnergyUnit",contactEnergyUnit.toString());
    //////	}
    //////}

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
    cerr << "size=" << size << endl;
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j) {

            cerr << "contact[" << i << "][" << j << "]=" << contactEnergyArray[i][j] << endl;

        }

    //Here I initialize max neighbor index for direct acces to the list of neighbors 
    boundaryStrategy = BoundaryStrategy::getInstance();
    maxNeighborIndex = 0;

    if (_xmlData->getFirstElement("WeightEnergyByDistance")) {
        weightDistance = true;
    }

    if (_xmlData->getFirstElement("Depth")) {
        maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromDepth(_xmlData->getFirstElement("Depth")->getDouble());
        //cerr<<"got here will do depth"<<endl;
    }
    else {
        //cerr<<"got here will do neighbor order"<<endl;
        if (_xmlData->getFirstElement("NeighborOrder")) {

            maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(_xmlData->getFirstElement("NeighborOrder")->getUInt());
            // 					exit(0);
        }
        else {
            maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);

        }

    }

    cerr << "Contact maxNeighborIndex=" << maxNeighborIndex << endl;

}

double ContactPlugin::changeEnergy(const Point3D &pt, const CellG *newCell, const CellG *oldCell) {

    //cerr<<"ChangeEnergy"<<endl;


    double energy = 0;
    unsigned int token = 0;
    double distance = 0;
    Point3D n;

    CellG *nCell = 0;
    WatchableField3D<CellG *> *fieldG = (WatchableField3D<CellG *> *) potts->getCellFieldG();
    Neighbor neighbor;



    if (weightDistance) {
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
                        energy -= contactEnergy(oldCell, nCell) / neighbor.distance;
                    }
                }
                else {
                    energy -= contactEnergy(oldCell, nCell) / neighbor.distance;
                }

            }
            if (nCell != newCell) {
                if ((newCell != 0) && (nCell != 0)) {
                    if ((newCell->clusterId) != (nCell->clusterId)) {
                        energy += contactEnergy(newCell, nCell) / neighbor.distance;
                    }
                }
                else {
                    energy += contactEnergy(newCell, nCell) / neighbor.distance;

                }
            }


        }
    }
    else {

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
                        energy -= contactEnergy(oldCell, nCell);
                    }
                }
                else {
                    energy -= contactEnergy(oldCell, nCell);
                }

            }
            if (nCell != newCell) {

                if ((newCell != 0) && (nCell != 0)) {
                    if ((newCell->clusterId) != (nCell->clusterId)) {
                        energy += contactEnergy(newCell, nCell);
                    }
                }
                else {
                    energy += contactEnergy(newCell, nCell);

                }

            }


        }


    }



    //cerr<<"pt="<<pt<<" energy="<<energy<<endl;
    //cerr<<"energy="<<energy<<endl;

    return energy;
}

double ContactPlugin::contactEnergy(const CellG *cell1, const CellG *cell2) {

    return contactEnergyArray[cell1 ? cell1->type : 0][cell2 ? cell2->type : 0];


}

void ContactPlugin::setContactEnergy(const string typeName1, const string typeName2, const double energy) {

    char type1 = automaton->getTypeId(typeName1);
    char type2 = automaton->getTypeId(typeName2);

    int index = getIndex(type1, type2);

    contactEnergies_t::iterator it = contactEnergies.find(index);
    ASSERT_OR_THROW(string("Contact energy for ") + typeName1 + " " + typeName2 +
        " already set!", it == contactEnergies.end());

    contactEnergies[index] = energy;
}

int ContactPlugin::getIndex(const int type1, const int type2) const {
    if (type1 < type2) return ((type1 + 1) | ((type2 + 1) << 16));
    else return ((type2 + 1) | ((type1 + 1) << 16));
}

std::string ContactPlugin::steerableName() { return "Contact"; }
std::string ContactPlugin::toString() { return steerableName(); }


