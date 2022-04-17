

#include "ChickGrowthTypePlugin.h"
#include "ChickGrowthNonCondensingTransition.h"
#include "ChickGrowthCondensingTransition.h"
#include <CompuCell3D/Automaton/CellType.h>
#include <CompuCell3D/ClassRegistry.h>
using namespace CompuCell3D;

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/Potts3D/Potts3D.h>
#include <CompuCell3D/steppables/PDESolvers/DiffusableVector.h>

#include <XMLCereal/XMLPullParser.h>
#include <XMLCereal/XMLSerializer.h>

#include <BasicUtils/BasicString.h>
#include <BasicUtils/BasicException.h>

#include <iostream>
using namespace std;

ChickGrowthTypePlugin::ChickGrowthTypePlugin() {}

ChickGrowthTypePlugin::~ChickGrowthTypePlugin() {}

void ChickGrowthTypePlugin::init(Simulator *simulator) {
           sim = simulator;
           potts = simulator->getPotts();
           potts->registerCellGChangeWatcher(this);
           potts->registerAutomaton(this);
           classType = new CellType();
           classType->addTransition(new ChickGrowthNonCondensingTransition(1));
           classType->addTransition(new ChickGrowthCondensingTransition(2));
}

float ChickGrowthTypePlugin::getConcentration(Point3D pt) {
    return
           ((DiffusableVector<float>*)sim->getClassRegistry()->getStepper(fieldSource))->getConcentrationField(fieldName)->get(pt);
}

unsigned char ChickGrowthTypePlugin::getCellType(const CellG *cell) const {
  if (!cell) return 0;
  else if (const_cast<CellG*>(cell)->type == 0)
      const_cast<CellG*>(cell)->type = 1;
  return const_cast<CellG*>(cell)->type; 
}

string ChickGrowthTypePlugin::getTypeName(const char type) const {
  switch (type) {
  case 0: return "Medium";
  case 1: return "NonCondensing";
  case 2: return "Condensing";
  default: THROW(string("Unknown cell type ") + BasicString(type) + "!");
  }
}

unsigned char ChickGrowthTypePlugin::getTypeId(const string typeName) const {
  if (typeName == "Medium") return 0;
  else if (typeName == "NonCondensing") return 1;
  else if (typeName == "Condensing") return 2;
  else THROW(string("Unknown cell type ") + typeName + "!");
}

void ChickGrowthTypePlugin::readXML(XMLPullParser &in) {
  in.skip(TEXT);

  while (in.check(START_ELEMENT)) {
    if (in.getName() == "Threshold") {
      threshold = BasicString::parseDouble(in.matchSimple());
    }
    else if (in.getName() == "ChemicalField") {
      fieldSource = in.getAttribute("Source").value;
      fieldName = in.getAttribute("Name").value;
      in.matchSimple();
    }
    else {
      throw BasicException(string("Unexpected element '") + in.getName() +
                           "'!", in.getLocation());
    }

    in.skip(TEXT);
  }
}

void ChickGrowthTypePlugin::writeXML(XMLSerializer &out) {
}
