#include <iostream>
#include <cmath>
#include <jni.h>
#include <vector>

using namespace std;
//using namespace CompuCell3D;

int main(){
	 JavaVM *jvm;

    JNIEnv *env;
    JavaVMInitArgs vm_args;

    JavaVMOption options;
    //Path to the java source code
    options.optionString = "-Djava.class.path=C:\\Program Files\\classes;C:\\JavaProjects\\Shapes\\bin";
    vm_args.version = JNI_VERSION_1_6; //JDK version. This indicates version 1.6
    vm_args.nOptions = 1;
    vm_args.options = &options;
    vm_args.ignoreUnrecognized = 0;

    int ret = JNI_CreateJavaVM(&jvm, (void**)&env, &vm_args);
	 cerr<<"return value="<<ret<<endl;
    if(ret < 0)
        printf("\nUnable to Launch JVM\n");
	 
	 cerr<<"jni version="<<std::hex<<env->GetVersion()<<endl;

	 jclass geometryClass=env->FindClass("Geometry/Square");
	 jclass stringClass=env->FindClass("java/lang/String");
	 cerr<<"geometryClass="<<geometryClass<<endl;
	 jmethodID constructorID=env->GetMethodID(geometryClass,"<init>","(D)V");
	 //jmethodID constructorID=env->GetStaticMethodID(geometryClass,"main","([Ljava/lang/String;)V");
	 jmethodID printNumberID=env->GetMethodID(geometryClass,"printNumber","()V");
	 
	 cerr<<"this is constructorID="<<constructorID<<endl;

	 
	 cerr<<"this is geometryClass="<<geometryClass<<endl;
	 cerr<<"this is stringClass="<<stringClass<<endl;
	 
	 //jobject squareObj=env->AllocObject(geometryClass);
	 jdouble squareSide=20;
	 vector<jobject> jobjectVec(10);
	 jobject squareObj=env->NewObject(geometryClass,constructorID,squareSide);
	 for (unsigned int i =0 ; i <jobjectVec.size() ; ++i){
		 squareSide=10+i;
		 jobjectVec[i]=env->NewObject(geometryClass,constructorID,squareSide);
	 }

	 for (unsigned int i =0 ; i <jobjectVec.size() ; ++i){
		 
		 env->CallObjectMethod(jobjectVec[i],printNumberID);
		 
	 }

	 return 0;
	 //jobject squareObj env->NewObject(geometryClass);
	 jmethodID methodID = env->GetStaticMethodID(geometryClass, "printNumberStatic", "()V");
	 cerr<<"this is method id "<<methodID<<endl;
	 env->CallStaticVoidMethod(geometryClass, methodID);

}