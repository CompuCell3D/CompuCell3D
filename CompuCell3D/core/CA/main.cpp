#include "CAManager.h"
#include "CACell.h"

#include <CompuCell3D/Field3D/Dim3D.h>
#include <CompuCell3D/Field3D/Point3D.h>

#include <BasicUtils/BasicException.h>

#include <iostream>

using namespace CompuCell3D;
using namespace std;


int main(){
	try{
		cerr<<"inside main"<<endl;
		CAManager caManager;
    
		caManager.createCellField(Dim3D(10,10,10));

		CACell *cell =  caManager.createAndPositionCell(Point3D(9,0,1));
    
		cerr<<"cell->type="<<(int)cell->type<<endl;

	}catch (BasicException & e){
		cerr<<e<<endl;
	}
    return 0;
}