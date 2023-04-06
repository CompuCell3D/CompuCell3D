#include <CGAL/basic.h>
#include <iostream>
#include <fstream>

#include <CGAL/Cartesian.h>
#include <CGAL/Segment_2.h>

typedef CGAL::Point_2 <CGAL::Cartesian<double>> Point;
typedef CGAL::Segment_2 <CGAL::Cartesian<double>> Segment;


#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/Potts3D/Potts3D.h>
#include <CompuCell3D/Field3D/Field3D.h>
#include <CompuCell3D/Field3D/WatchableField3D.h>
#include <CompuCell3D/Boundary/BoundaryStrategy.h>

#include <CompuCell3D/Potts3D/CellInventory.h>
#include <CompuCell3D/Automaton/Automaton.h>
#include <BasicUtils/BasicString.h>
#include <BasicUtils/BasicException.h>
#include <PublicUtilities/StringUtils.h>
#include <algorithm>

using namespace CompuCell3D;


#include <iostream>

using namespace std;

#include "CGALMeshDumper.h"
#include <Logger/CC3DLogger.h>

CGALMeshDumper::CGALMeshDumper() : cellFieldG(0), sim(0), potts(0), xmlData(0), boundaryStrategy(0), automaton(0),
                                   cellInventoryPtr(0) {}

CGALMeshDumper::~CGALMeshDumper() {
}


void CGALMeshDumper::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
    xmlData = _xmlData;

    potts = simulator->getPotts();
    cellInventoryPtr = &potts->getCellInventory();
    sim = simulator;
    cellFieldG = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();
    fieldDim = cellFieldG->getDim();


    simulator->registerSteerableObject(this);

    update(_xmlData, true);

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CGALMeshDumper::extraInit(Simulator *simulator) {
    //PUT YOUR CODE HERE
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CGALMeshDumper::start() {

    //PUT YOUR CODE HERE

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CGALMeshDumper::step(const unsigned int currentStep) {
    //REPLACE SAMPLE CODE BELOW WITH YOUR OWN
	CellInventory::cellInventoryIterator cInvItr;
	CellG * cell;
    CC3D_Log(LOG_DEBUG) << "currentStep="<<currentStep;
    for (cInvItr = cellInventoryPtr->cellInventoryBegin(); cInvItr != cellInventoryPtr->cellInventoryEnd(); ++cInvItr) {
        cell = cellInventoryPtr->getCell(cInvItr);
        CC3D_Log(LOG_DEBUG) << "cell.id="<<cell->id<<" vol="<<cell->volume;
    }

}


void CGALMeshDumper::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {

    //PARSE XML IN THIS FUNCTION
    //For more information on XML parser function please see CC3D code or lookup XML utils API
    automaton = potts->getAutomaton();
    ASSERT_OR_THROW(
            "CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET",
            automaton)
    set<unsigned char> cellTypesSet;

    CC3DXMLElement *exampleXMLElem = _xmlData->getFirstElement("Example");
    if (exampleXMLElem) {
        double param = exampleXMLElem->getDouble();
        CC3D_Log(LOG_DEBUG) << "param="<<param;
        if (exampleXMLElem->findAttribute("Type")) {
            std::string attrib = exampleXMLElem->getAttribute("Type");
            // double attrib=exampleXMLElem->getAttributeAsDouble("Type"); //in case attribute is of type double
            CC3D_Log(LOG_DEBUG) << "attrib="<<attrib;
        }
    }

    //boundaryStrategy has information aobut pixel neighbors 
    boundaryStrategy = BoundaryStrategy::getInstance();

}

std::string CGALMeshDumper::toString() {
    return "CGALMeshDumper";
}

std::string CGALMeshDumper::steerableName() {
    return toString();
}
        
