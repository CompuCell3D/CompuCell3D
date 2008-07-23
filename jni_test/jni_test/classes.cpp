#include <iostream>
#include <cmath>
#include "classes.h"

#define EXP_STL

using namespace std;
using namespace CompuCell3D;

Calculator::Calculator(){
  cerr<<"THIS IS INSIDE CONSTRUCTOR"<<endl;
}
Calculator::~Calculator(){
   cerr<<"THIS IS INSIDE Destructor"<<endl;
	
}

void Calculator::calculate(int _max){

     for (int i =0 ; i <_max; ++i){
         cerr<<"cos("<<i<<")="<<cos((double)i)<<endl;


     }
}


CalculatorJava::CalculatorJava(){
  cerr<<"THIS IS INSIDE CONSTRUCTOR CALCULATOR JAVA"<<endl;
}
CalculatorJava::~CalculatorJava(){

   cerr<<"THIS IS INSIDE Destructor Calculator Java"<<endl;
	env->DeleteGlobalRef(shapeObjExt);
	env->DeleteGlobalRef(shapeObjEx1);
	
}

void CalculatorJava::calculate(int _max){

     for (int i =0 ; i <_max; ++i){
         cerr<<"JAVA CALCULATORcos("<<i<<")="<<cos((double)i)<<endl;


     }
}

void CalculatorJava::calculate1(int _max){

     for (int i =0 ; i <_max; ++i){
         cerr<<"1 JAVA CALCULATOR cos("<<i<<")="<<cos((double)i)<<endl;


     }
}



jobject CalculatorJava::getRefShapeObj(){

		return shapeObj;
}

void CalculatorJava::setShapeObject(jobject _extObj){
	
	shapeObjExt=env->NewGlobalRef(_extObj);
   //shapeObjExt=_extObj;
	cerr<<"SETTING shapeObjExt="<<shapeObjExt<<endl;

}

void CalculatorJava::setShapeObject1(jobject _extObj){
	
	shapeObjExt1=env->NewGlobalRef(_extObj);
	//shapeObjExt1=_extObj;
	cerr<<"SETTING shapeObjExt1="<<shapeObjExt1<<endl;
}

void CalculatorJava::runShapeObject(jobject  _extObj1){

		jclass geometryClass=env->FindClass("Geometry/Square");
	   
		jmethodID printNumberID=env->GetMethodID(geometryClass,"printNumber","()V");
		env->CallVoidMethod(_extObj1,printNumberID);
		cerr<<"_extObj="<<_extObj1<<endl;
		cerr<<"*******************************************************"<<endl;
		jmethodID printNumberID1=env->GetMethodID(geometryClass,"printNumber","()V");
		cerr<<"shapeObjExt="<<shapeObjExt<<endl;
		env->CallVoidMethod(shapeObjExt,printNumberID1);
}

void CalculatorJava::runShapeObjects(){
	jclass geometryClass=env->FindClass("Geometry/Square");
	jmethodID printNumberID=env->GetMethodID(geometryClass,"printNumber","()V");
	cerr<<"BEFORE env->CallVoidMethod(shapeObjExt,printNumberID)"<<endl;
	env->CallVoidMethod(shapeObjExt,printNumberID);
	cerr<<"AFTER env->CallVoidMethod(shapeObjExt,printNumberID)"<<endl;

	env->CallVoidMethod(shapeObjExt1,printNumberID);

}

 void 
	 CalculatorJava::callJavaFunction(){
		 JavaVMInitArgs vm_args; 
		
		jint res = 0;
		char *classpath,*librarypath;
		vms=NULL;
		jsize no_vms;

		res = JNI_GetCreatedJavaVMs(&vms, 1, &no_vms);
		if(res == 0 && no_vms == 0)
		{
			cerr<<"got "<<res<<" Virtual machines"<<endl;
			//res = JNI_CreateJavaVM(&jvm,(void **)&env,&vm_args);
		}else{
			cerr<<"got "<<no_vms<<" Virtual machines"<<endl;
		}
		jint jni_version=JNI_VERSION_1_6;
		cerr<<"vms ptr="<<vms<<endl;
		//vms->JNI_GetCreatedJavaVMs(&vm_args);
		
	   vms->GetEnv((void**)&env, jni_version);

		cerr<<"jni version="<<std::hex<<env->GetVersion()<<endl;
		
		jclass geometryClass=env->FindClass("Geometry/Square");
	   jmethodID constructorID=env->GetMethodID(geometryClass,"<init>","(D)V");
		//jmethodID printNumberID=env->GetMethodID(geometryClass,"printNumber","()V");
		
		//cerr<<"THIS IS NEW OBJECT BEFORE INITIALIZATION="<<shapeObj<<endl;
		objectWrapperPtr=new JavaObjectWrapper();
		jdouble squareSideWrap=21.21;
		objectWrapperPtr->shapeObjWrap=env->NewObject(geometryClass,constructorID,squareSideWrap);
		wrapper.shapeObjWrap=env->NewObject(geometryClass,constructorID,squareSideWrap);

		jdouble squareSide=21;
		shapeObj=env->NewObject(geometryClass,constructorID,squareSide);

		jdouble squareSide1=20.201;
		shapeObj1=env->NewObject(geometryClass,constructorID,squareSide1);


		jclass dictionaryWrapper=env->FindClass("Geometry/JDictWrapper");
		cerr<<"dictionaryWrapper="<<dictionaryWrapper<<endl;
		jmethodID constructorDictionaryWrapperID=env->GetMethodID(dictionaryWrapper,"<init>","()V");
		cerr<<"constructorDictionaryWrapperID="<<constructorDictionaryWrapperID<<endl;
		pyDict=env->NewObject(dictionaryWrapper,constructorDictionaryWrapperID);


		jclass listWrapper=env->FindClass("Geometry/JListWrapper");
		cerr<<"listWrapper="<<listWrapper<<endl;
		jmethodID constructorListWrapperID=env->GetMethodID(listWrapper,"<init>","()V");
		cerr<<"constructorListWrapperID="<<constructorListWrapperID<<endl;
		pyList=env->NewObject(listWrapper,constructorListWrapperID);




		

		//jclass geometryClass=env->FindClass("java/lang/String");
		cerr<<"THIS IS geometryClass="<<geometryClass<<endl;
		jmethodID methodID = env->GetStaticMethodID(geometryClass, "printNumberStatic", "()V");

		env->CallStaticVoidMethod(geometryClass, methodID);
		return;

  //  JNIEnv *env;
  //  JavaVMInitArgs vm_args;

  //  JavaVMOption options; 
  //  //Path to the java source code     
  //  options.optionString = "-Djava.class.path=C:\\Program Files\\classes\CalculatorTry"; 
  //  vm_args.version	= JNI_VERSION_1_6; //JDK version. This indicates version 1.6
  //  vm_args.nOptions = 1;
  //  vm_args.options = &options;
  //  vm_args.ignoreUnrecognized = 0;
  //  
  //  int ret = JNI_CreateJavaVM(&jvm, (void**)&env, &vm_args);
	 //cerr<<"return value="<<ret<<endl;
  //  if(ret < 0)
  //      printf("\nUnable to Launch JVM\n");       
    

  //  JavaVMOption options[2];
  //  JavaVMInitArgs vm_args;
  //long result;
  //

  //options[0].optionString = ".";
  //options[1].optionString = "-Djava.compiler=NONE";

  //vm_args.version = JNI_VERSION_1_6;
  //vm_args.options = options;
  //vm_args.nOptions = 2;
  //vm_args.ignoreUnrecognized = JNI_FALSE;

  //result = JNI_CreateJavaVM(
  //           &jvm, (void**)&jenv, &vm_args);
  //if(result == JNI_ERR ) {
  //  printf("Error invoking the JVM");
  //  exit (-1);
  //}
	
		//jclass geometryClass=jenv->GetObjectClass(_jclass);
		//jmethodID printNumberID=jenv->GetMethodID(geometryClass,"printNumber","(V)V");
		//jenv->CallVoidMethod(geometryClass,printNumberID);


		//const char* filePath = _jenv->GetStringUTFChars(jpath, 0); 
		//return 0;
 }


//void CalculatorJava::callJavaFunction(){
//	jclass myClass=((void**)jenv)->GetObjectClass(jenv,geometryObject);
//
//}

