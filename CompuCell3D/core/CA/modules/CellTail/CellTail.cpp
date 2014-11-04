

#include <CA/CACell.h> 
#include <CA/CACellStack.h> 
#include <CA/CAManager.h> 


using namespace CompuCell3D;


using namespace std;


#include "CellTail.h"
#include <iostream>

CellTail::CellTail():boundaryStrategy(0),caManager(0),cellField(0) {}

CellTail::~CellTail() {}


void CellTail::init(CAManager *_caManager){
	RUNTIME_ASSERT_OR_THROW("CellTail::init _caManager cannot be NULL!",_caManager);
	caManager=_caManager;
	
	

}
void CellTail::field3DChange(CACell *_movingCell, CACellStack *_sourceCellStack,CACellStack *_targetCellStack){

}
//void CellTail::field3DChange(CACell *_movingCell, CACellStack *_sourceCellStack,CACellStack *_targetCellStack){
//    cerr<<"moving cell ="<<_movingCell<<" id="<<_movingCell->id<<" source pt="<<_sourceCellStack->getLocation()<<" target pt="<<_targetCellStack->getLocation()<<endl;
//}  


std::string CellTail::toString(){
    return "CellTail";
}