#include <iostream>
#include <cmath>
#include "dllDeclarationSpecifier.h"
#include <jni.h>

namespace CompuCell3D{
class DECLSPECIFIER Calculator{
 public:
    Calculator();
    ~Calculator();
    void calculate(int _max);

};

class DECLSPECIFIER CalculatorJava{
 public:
    CalculatorJava();
    ~CalculatorJava();
    void calculate(int _max);
	 JNIEnv *jenv;
	 JavaVM *jvm;
	 jobject geometryObject;
	 void callJavaFunction();
	 void calculate1(int _max);	 
	 jobject getRefShapeObj();
	 jobject shapeObj;
	 jobject pyDict;
	 jobject pyList;
	 


};


};