

#include <CA/CACell.h> 
#include <CA/CACellStack.h> 
#include <CA/CAManager.h> 


using namespace CompuCell3D;


using namespace std;


#include "CellTrail.h"
#include <iostream>
//////////////////////////////////////////////////////////////////////////////////////////
CellTrail::CellTrail():boundaryStrategy(0),caManager(0),cellField(0) {}
//////////////////////////////////////////////////////////////////////////////////////////
CellTrail::~CellTrail() {}
//////////////////////////////////////////////////////////////////////////////////////////

void CellTrail::init(CAManager *_caManager){
	RUNTIME_ASSERT_OR_THROW("CellTrail::init _caManager cannot be NULL!",_caManager);
	caManager=_caManager;		
}
//////////////////////////////////////////////////////////////////////////////////////////
void CellTrail::extraInit(){
}


//////////////////////////////////////////////////////////////////////////////////////////
void CellTrail::_addMovingCellTrail(std::string _movingCellType, std::string _trailCellType, int _trailCellSize){
	unsigned char movingCellTypeId = caManager->getTypeId(_movingCellType);
	unsigned char trailCellTypeId = caManager->getTypeId(_trailCellType);

	movingTypeId2TrailTypeIdMap[movingCellTypeId]=make_pair(trailCellTypeId,_trailCellSize);

}
//////////////////////////////////////////////////////////////////////////////////////////
void CellTrail::field3DChange(CACell *_movingCell, CACellStack *_sourceCellStack,CACellStack *_targetCellStack){
	 if (! _sourceCellStack || ! _targetCellStack) 	return; //we only tun this is target and source stacks are non-zero
	 //cerr<<"moving cell ="<<_movingCell<<" id="<<_movingCell->id<<" source pt="<<_sourceCellStack<<" target pt="<<_targetCellStack<<endl;
	 //cerr<<"moving cell ="<<_movingCell<<" id="<<_movingCell->id<<" source pt="<<_sourceCellStack->getLocation()<<" target pt="<<_targetCellStack->getLocation()<<endl;
	 mitr_t mitr = movingTypeId2TrailTypeIdMap.find(_movingCell->type);
	 if (mitr !=  movingTypeId2TrailTypeIdMap.end() ){
		 int trailCellSize = mitr->second.second;
		 int trailCellType = mitr->second.first;
		 if (_sourceCellStack->getCapacity() - _sourceCellStack->getFillLevel() >= trailCellSize){

			 bool createNewCellFlag=true;
			 for (int i =  0 ; i < _sourceCellStack->getNumCells() ; ++i){
				 CACell * cellS = _sourceCellStack->getCellByIdx(i);
				 if (cellS->type == trailCellType ){
					 createNewCellFlag=false; //we do not create tail if the cell like that already exists
					 break;
				 }
			 }
			 if (createNewCellFlag){
				 if (_sourceCellStack->canFit(trailCellSize)){
					 CACell * cell = caManager->createCell();
					 cell -> type = mitr->second.first;
					 cerr<<"leaving trail with cell type="<<(int)cell -> type <<endl;
					 _sourceCellStack ->appendCell(cell);
				 }
			 }
		 }
		 
		 
		 
		 
		 
	 }

}
//void CellTail::field3DChange(CACell *_movingCell, CACellStack *_sourceCellStack,CACellStack *_targetCellStack){
//    cerr<<"moving cell ="<<_movingCell<<" id="<<_movingCell->id<<" source pt="<<_sourceCellStack->getLocation()<<" target pt="<<_targetCellStack->getLocation()<<endl;
//}  

//////////////////////////////////////////////////////////////////////////////////////////
std::string CellTrail::toString(){
    return "CellTrail";
}