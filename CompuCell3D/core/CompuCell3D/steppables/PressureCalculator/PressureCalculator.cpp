



#include <CompuCell3D/CC3D.h>



using namespace CompuCell3D;



using namespace std;



#include "PressureCalculator.h"



PressureCalculator::PressureCalculator() : cellFieldG(0),sim(0),potts(0),xmlData(0),boundaryStrategy(0),automaton(0),cellInventoryPtr(0){}



PressureCalculator::~PressureCalculator() {

}





void PressureCalculator::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

  xmlData=_xmlData;

  

  potts = simulator->getPotts();

  cellInventoryPtr=& potts->getCellInventory();

  sim=simulator;

  cellFieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();

  fieldDim=cellFieldG->getDim();



  potts->getCellFactoryGroupPtr()->registerClass(&pressureCalculatorDataAccessor);

  simulator->registerSteerableObject(this);



  update(_xmlData,true);



}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



void PressureCalculator::extraInit(Simulator *simulator){

    //PUT YOUR CODE HERE

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////





//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PressureCalculator::start(){



  //PUT YOUR CODE HERE



}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



void PressureCalculator::step(const unsigned int currentStep){

    //REPLACE SAMPLE CODE BELOW WITH YOUR OWN

	CellInventory::cellInventoryIterator cInvItr;

	CellG * cell=0;

    

    cerr<<"currentStep="<<currentStep<<endl;

	for(cInvItr=cellInventoryPtr->cellInventoryBegin() ; cInvItr !=cellInventoryPtr->cellInventoryEnd() ;++cInvItr )

	{

		cell=cellInventoryPtr->getCell(cInvItr);

        cerr<<"cell.id="<<cell->id<<" vol="<<cell->volume<<endl;

    }



}





void PressureCalculator::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){



    //PARSE XML IN THIS FUNCTION

    //For more information on XML parser function please see CC3D code or lookup XML utils API

    automaton = potts->getAutomaton();

    ASSERT_OR_THROW("CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET", automaton)

   set<unsigned char> cellTypesSet;



    CC3DXMLElement * exampleXMLElem=_xmlData->getFirstElement("Example");

    if (exampleXMLElem){

        double param=exampleXMLElem->getDouble();

        cerr<<"param="<<param<<endl;

        if(exampleXMLElem->findAttribute("Type")){

            std::string attrib=exampleXMLElem->getAttribute("Type");

            // double attrib=exampleXMLElem->getAttributeAsDouble("Type"); //in case attribute is of type double

            cerr<<"attrib="<<attrib<<endl;

        }

    }

    

    //boundaryStrategy has information aobut pixel neighbors 

    boundaryStrategy=BoundaryStrategy::getInstance();



}



std::string PressureCalculator::toString(){

   return "PressureCalculator";

}



std::string PressureCalculator::steerableName(){

   return toString();

}

        

