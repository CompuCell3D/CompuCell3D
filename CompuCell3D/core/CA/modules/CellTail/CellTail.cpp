

#include <CA/CACell.h> 
#include <CA/CACellStack.h> 
#include <CA/CAManager.h> 


using namespace CompuCell3D;


using namespace std;


#include "CellTail.h"
#include <iostream>
//////////////////////////////////////////////////////////////////////////////////////////
CellTail::CellTail():boundaryStrategy(0),caManager(0),cellField(0) {}
//////////////////////////////////////////////////////////////////////////////////////////
CellTail::~CellTail() {}
//////////////////////////////////////////////////////////////////////////////////////////

void CellTail::init(CAManager *_caManager){
	RUNTIME_ASSERT_OR_THROW("CellTail::init _caManager cannot be NULL!",_caManager);
	caManager=_caManager;		
}
//////////////////////////////////////////////////////////////////////////////////////////
void CellTail::extraInit(){
}


//////////////////////////////////////////////////////////////////////////////////////////
void CellTail::setMovingCellTrail(std::string _movingCellType, std::string _tailCellType, int _tailCellSize){
	unsigned char movingCellTypeId = caManager->getTypeId(_movingCellType);
	unsigned char tailCellTypeId = caManager->getTypeId(_tailCellType);

	movingTypeId2TailTypeIdMap[movingCellTypeId]=make_pair(tailCellTypeId,_tailCellSize);

}
//////////////////////////////////////////////////////////////////////////////////////////
void CellTail::field3DChange(CACell *_movingCell, CACellStack *_sourceCellStack,CACellStack *_targetCellStack){
	 if (! _sourceCellStack || ! _targetCellStack) 	return; //we only tun this is target and source stacks are non-zero
	 //cerr<<"moving cell ="<<_movingCell<<" id="<<_movingCell->id<<" source pt="<<_sourceCellStack<<" target pt="<<_targetCellStack<<endl;
	 //cerr<<"moving cell ="<<_movingCell<<" id="<<_movingCell->id<<" source pt="<<_sourceCellStack->getLocation()<<" target pt="<<_targetCellStack->getLocation()<<endl;
	 mitr_t mitr = movingTypeId2TailTypeIdMap.find(_movingCell->type);
	 if (mitr !=  movingTypeId2TailTypeIdMap.end() ){
		 int tailCellSize = mitr->second.second;
		 int tailCellType = mitr->second.first;
		 if (_sourceCellStack->getCapacity() - _sourceCellStack->getFillLevel() >= tailCellSize){

			 bool createNewCellFlag=true;
			 for (int i =  0 ; i < _sourceCellStack->getNumCells() ; ++i){
				 CACell * cellS = _sourceCellStack->getCellByIdx(i);
				 if (cellS->type == tailCellType ){
					 createNewCellFlag=false; //we do not create tail if the cell like that already exists
					 break;
				 }
			 }
			 if (createNewCellFlag){
				 if (_sourceCellStack->canFit(tailCellSize)){
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
std::string CellTail::toString(){
    return "CellTail";
}