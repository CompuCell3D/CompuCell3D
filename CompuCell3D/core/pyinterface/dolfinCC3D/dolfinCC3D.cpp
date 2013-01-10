#include "dolfinCC3D.h"
#include <iostream>
using namespace std;

dolfinCC3D::~dolfinCC3D(){
   cerr<<"destructor of dolfinCC3D"<<endl;
}

double dolfinCC3D::getX(){
   cerr<<"This is value of x "<<x<<endl;
   return x;
}