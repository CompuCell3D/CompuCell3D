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

class DECLSPECIFIER JavaObjectWrapper{
public:
	jobject shapeObjWrap;
};


class DECLSPECIFIER CalculatorJava{
 public:
    CalculatorJava();
    ~CalculatorJava();
    void calculate(int _max);
	 JNIEnv *jenv;
	 JavaVM *jvm;
	 
	 JNIEnv *env;
	 JavaVM *vms;

	 jobject geometryObject;
	 void callJavaFunction();
	 void calculate1(int _max);	 
	 jobject getRefShapeObj();
	 jobject shapeObj;
	 jobject shapeObj1;
	 jobject pyDict;
	 jobject pyList;
	 JavaObjectWrapper * objectWrapperPtr;
	 JavaObjectWrapper wrapper;
	 void setShapeObject(jobject  _extObj);
	 void setShapeObject1(jobject  _extObj);
	 void runShapeObject(jobject  _extObj1);
	 void runShapeObjects();

	 jobject  shapeObjExt;
	 jobject  shapeObjExt1;
	 


};


};