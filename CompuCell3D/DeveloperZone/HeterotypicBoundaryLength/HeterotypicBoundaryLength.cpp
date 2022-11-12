



#include <CompuCell3D/CC3D.h>
#include <Logger/CC3DLogger.h>


using namespace CompuCell3D;



using namespace std;



#include "HeterotypicBoundaryLength.h"



HeterotypicBoundaryLength::HeterotypicBoundaryLength() : cellFieldG(0),sim(0),potts(0),xmlData(0),boundaryStrategy(0),automaton(0),cellInventoryPtr(0){}



HeterotypicBoundaryLength::~HeterotypicBoundaryLength() {

}

unsigned int HeterotypicBoundaryLength::typePairIndex(unsigned int cellType1, unsigned int cellType2) {
    return 256 * cellType2 + cellType1;

}

void HeterotypicBoundaryLength::calculateHeterotypicSurface() {

    unsigned int maxNeighborIndex = this->boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);
    Neighbor neighbor;

    CellG *nCell = 0;

    this->typePairHTSurfaceMap.clear();

    // note: unit surface is different on a hex lattice. if you are runnign 
    // this steppable on hex lattice you need to adjust it. Remember that on hex lattice unit length and unit surface have different values
    double unit_surface = 1.0; 
    CC3D_Log(LOG_DEBUG) <<  "Calculating HTBL for all cell type combinations";

    for (unsigned int x = 0; x < fieldDim.x; ++x)
        for (unsigned int y = 0; y < fieldDim.y; ++y)
            for (unsigned int z = 0; z < fieldDim.z; ++z) {
                Point3D pt(x, y, z);
                CellG *cell = this->cellFieldG->get(pt);

                unsigned int cell_type = 0;
                if (cell) {
                    cell_type = (unsigned int)cell->type;
                }

                for (unsigned int nIdx = 0; nIdx <= maxNeighborIndex; ++nIdx) {
                    neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt), nIdx);
                    if (!neighbor.distance) {
                        //if distance is 0 then the neighbor returned is invalid
                        continue;
                    }

                    nCell = this->cellFieldG->get(neighbor.pt);
                    unsigned int n_cell_type = 0;
                    if (nCell) {
                        n_cell_type = (unsigned int)nCell->type;
                    }


                    if (nCell != cell) {                        
                        unsigned int pair_index_1 = typePairIndex(cell_type, n_cell_type);
                        unsigned int pair_index_2 = typePairIndex(n_cell_type, cell_type);
                        this->typePairHTSurfaceMap[pair_index_1] += unit_surface; 
                        if (pair_index_1 != pair_index_2) {
                            this->typePairHTSurfaceMap[pair_index_2] += unit_surface;
                        }
                        
                    }

                }

            }

}

double HeterotypicBoundaryLength::getHeterotypicSurface(unsigned int cellType1, unsigned int cellType2) {
    unsigned int pair_index = typePairIndex(cellType1, cellType2);
    
    double heterotypic_surface = this->typePairHTSurfaceMap[pair_index]/2.0;
    
    return heterotypic_surface;
}


void HeterotypicBoundaryLength::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

  xmlData=_xmlData;

  

  potts = simulator->getPotts();

  cellInventoryPtr=& potts->getCellInventory();

  sim=simulator;

  cellFieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();

  fieldDim=cellFieldG->getDim();



  

  simulator->registerSteerableObject(this);



  update(_xmlData,true);



}


void HeterotypicBoundaryLength::extraInit(Simulator *simulator){

}


void HeterotypicBoundaryLength::start(){

}


void HeterotypicBoundaryLength::step(const unsigned int currentStep){

}


void HeterotypicBoundaryLength::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){



    //PARSE XML IN THIS FUNCTION

    //For more information on XML parser function please see CC3D code or lookup XML utils API

    automaton = potts->getAutomaton();

    ASSERT_OR_THROW("CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET", automaton)
  
    //boundaryStrategy has information aobut pixel neighbors 
    boundaryStrategy=BoundaryStrategy::getInstance();

}

std::string HeterotypicBoundaryLength::toString(){
   return "HeterotypicBoundaryLength";
}

std::string HeterotypicBoundaryLength::steerableName(){

   return toString();

}

        

