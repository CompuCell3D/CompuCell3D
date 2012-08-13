#include <iostream>


#include "simthreadAccessor.h"

using namespace std;

SimthreadBase * getSimthreadBasePtr(){
    cerr<<"THIS IS RETURNED simthreadBasePtr "<<simthreadBasePtr<<endl;
	return simthreadBasePtr;

}

double getNumberGlobal(){return numberGlobal;}

void printLibName(){cerr<<"This is simthreadAccessor module"<<endl;}