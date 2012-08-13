#include "PyNewPlugin.h"
#include <iostream>
using namespace std;

PyNewPlugin::~PyNewPlugin(){
   cerr<<"destructor of PyNewPlugin"<<endl;
}

double PyNewPlugin::getX(){
   cerr<<"This is value of x "<<x<<endl;
   return x;
}