
#ifndef CACELLSTACK_H
#define CACELLSTACK_H


#include <vector>
#include <map>
#include <CompuCell3D/Field3D/Point3D.h>
#include "CADLLSpecifier.h"

namespace CompuCell3D {

   class CACell;

   class CASHARED_EXPORT CACellStack{
	   //std::vector<CACell *> stack;
	   std::map<long, CACell *> stack;
	   int fillLevel;
	   int capacity;
	   Point3D pt;
   public:
	   CACellStack(int _capacity=0,Point3D _pt=Point3D(0,0,0));
	   ~CACellStack();
	   Point3D getLocation(){return pt;}
	   int getFillLevel(){return fillLevel;}
	   int getNumCells(){return stack.size();}		
	   bool isFull(){return capacity <= fillLevel;}
	   CACell * appendCell(CACell * _cell);
	   CACell * forceAppendCell(CACell * _cell); //returns ptr of the cell that was removed from stack

	   void deleteCell(CACell * _cell);
	   CACell * getCellByIdx(int _idx);
   	
   };

};
#endif
