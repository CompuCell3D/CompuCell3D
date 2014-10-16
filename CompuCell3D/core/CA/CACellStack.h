
#ifndef CACELLSTACK_H
#define CACELLSTACK_H


#include <vector>
#include <map>
#include "CADLLSpecifier.h"

namespace CompuCell3D {

   class CACell;

   class CASHARED_EXPORT CACellStack{
	   //std::vector<CACell *> stack;
	   std::map<long, CACell *> stack;
	   int fillLevel;
	   int capacity;
   public:
	   CACellStack(int _capacity=0);
	   ~CACellStack();
	   int getFillLevel(){return stack.size();}
	   bool isFull(){return capacity==stack.size();}
	   bool appendCell(CACell * _cell);
	   CACell * appendCellForce(CACell * _cell); //returns ptr of the cell that was removed from stack

	   void deleteCell(CACell * _cell);
	   CACell * getCellByIdx(int _idx);
   	
   };

};
#endif
