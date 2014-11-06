#include "CACellStack.h"
#include "CACell.h"
#include <iostream>


#include <limits>

#undef max
#undef min

//#define _DEBUG

using namespace std;

namespace CompuCell3D {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CACellStack::CACellStack(int _capacity,Point3D _pt):
fillLevel(0),
capacity(_capacity),
pt(_pt)
{
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CACellStack::~CACellStack(){

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool CACellStack::appendCell(CACell * _cell){
	if (! canFit(_cell->size) ) return false; //there is no place in the stack
	if (stack.find(_cell->id) != stack.end()) return false; //cell with this id is already in the stack

	_cell->xCOM=pt.x;
	_cell->yCOM=pt.y;
	_cell->zCOM=pt.z;

	stack[_cell->id]=_cell;
	fillLevel += _cell->size;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//CACell * CACellStack::forceAppendCell(CACell * _cell){
//	CACell * removedCell=0;
//#ifdef _DEBUG
//	cerr<<"CACellStack capacity="<<capacity<<endl;
//	cerr<<"isFull()="<<isFull()<<endl;
//#endif
//	if (isFull()){
//		//removing last entry
//		std::map<long,CACell *>::iterator mitr=stack.end();
//		--mitr;
//		removedCell=mitr->second;
//
//		stack.erase(mitr);
//
//		fillLevel -= removedCell->size;
//	}
//
//	appendCell(_cell);
//
//	fillLevel += _cell->size;
//
//	return removedCell; 
//}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CACellStack::removeCell(CACell * _cell){
	
	
	std::map<long,CACell *>::iterator mitr=stack.find(_cell->id);
	if (mitr != stack.end()){
		stack.erase(mitr);
		fillLevel -= mitr->second->size;
	}
	
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CACell * CACellStack::getCellByIdx(int _idx){
	//no bound checking in this function...
	std::map<long,CACell *>::iterator mitr=stack.begin();
	advance(mitr,_idx);
	return mitr->second;

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}