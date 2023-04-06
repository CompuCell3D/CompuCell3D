#include "ReactionKernels.h"


#include <fstream>
#include <sstream>
#include <cassert>
#include <CompuCell3D/CC3DExceptions.h>

using namespace std;
using namespace CompuCell3D;

string getTempDir(){
    char *res=getenv("TMP");
    if(res)
        return res;

    res=getenv("TEMP");
    if(res)
        return res;

    res=getenv("TMPDIR");
    if(res)
        return res;

    ASSERT_OR_THROW("Can not detect temporary directory, check getTempDir function", false);
}

std::string getTempFileName(){
    char *buff=tempnam(getTempDir().c_str(), "clCC3D");
    ASSERT_OR_THROW("Can not generate temprary file name for OpenCL kernels", buff);
    return buff;
}

std::string FieldName(fieldNameAddTerm_t const &fnat){
    return fnat.first;
}

std::string AdditionalTerm(fieldNameAddTerm_t const &fnat){
    return fnat.second;
}

std::string genReactionKernelFunc(std::ofstream &f, size_t ind, std::vector<fieldNameAddTerm_t> const &fieldNameAddTerms){

    assert(ind<fieldNameAddTerms.size());

    f<<std::endl;
    f<<"inline"<<endl;

    std::stringstream funcName;
    funcName<<"AdditionalTerm"<<FieldName(fieldNameAddTerms[ind]);
    f<<"float "<<funcName.str()<<"(";
    for(size_t i=0; i<fieldNameAddTerms.size(); ++i){
        f<<"float "<<FieldName(fieldNameAddTerms[i]);
        if(i!=fieldNameAddTerms.size()-1)
            f<<", ";
    }

    f<<"){"<<endl;

    //f<<"    return ";
    f<<"\t"<<AdditionalTerm(fieldNameAddTerms[ind])<<endl;
    f<<"}"<<endl;
    return funcName.str();
}

string genReactionKernels(std::vector<fieldNameAddTerm_t> const &fieldNameAddTerms){

    string tmpKernelFileName=getTempFileName();

    std::cout<<"Temporary file generated for reaction term kernels: \n\t"<<tmpKernelFileName<<std::endl;
    std::ofstream f(tmpKernelFileName.c_str());
    ASSERT_OR_THROW("Can not open generated file for writing", f.is_open());

    std::vector<std::string> funcNames;
    for(size_t i=0; i<fieldNameAddTerms.size(); ++i){
        //funcNames.push_back(genReactionKernelFunc(f, i));
        funcNames.push_back(genReactionKernelFunc(f, i, fieldNameAddTerms));
    }

    {
        f<<endl<<"__kernel void minusG(float dt,"<<endl<<
         "\t__global const float* g_prevField,"<<endl<<
         "\t__global const float* g_newField,"<<endl<<
         "\tint3 dim,"<<endl<<
         "\t__global float * g_result,"<<endl<<
         "\tint stride)"<<endl<<
         "{"<<endl;

        f<<"\tint4 ind3d={get_global_id(0), get_global_id(1), get_global_id(2), 0};"<<endl;

        f<<"\tif(ind3d.x>=dim.x||ind3d.y>=dim.y||ind3d.z>=dim.z)"<<endl<<
         "\t\treturn;"<<endl;

        f<<"\tsize_t ind=d3To1d(ind3d, dim);"<<endl;

        for(size_t i=0; i<fieldNameAddTerms.size(); ++i){
            f<<"\tfloat field"<<i<<"=(g_newField+"<<"stride*"<<i<<")[ind];"<<endl;
        }

        //making a parameter list. It is the same for all reaction functions so far
        stringstream arg;
        for(size_t i=0; i<fieldNameAddTerms.size(); ++i){
            arg<<"field"<<i;
            if(i!=fieldNameAddTerms.size()-1){
                arg<<", ";
            }
        }

        for(size_t i=0; i<fieldNameAddTerms.size(); ++i){
            f<<"\t(g_result+stride*"<<i<<")[ind]="<<
             "(g_prevField+stride*"<<i<<")[ind]-(g_result+stride*"<<i<<")[ind]+"<<"dt*"<<funcNames[i]<<"("<<arg.str()<<");"<<endl;
        }

        //f<<"\t(g_result+stride*"<<0<<")[ind]=(g_prevField+stride*"<<0<<")[ind];"<<endl;

        f<<"}"<<endl;
    }


    {
        f<<endl<<"__kernel void Jacobian(float dt,"<<endl<<
         "\tfloat epsilon,"<<endl<<
         "\t__global const float* g_newField,"<<endl<<
         "\t__global const float* g_update,"<<endl<<
         "\tint3 dim,"<<endl<<
         "\t__global float * g_result,"<<endl<<
         "\tint stride)"<<endl<<
         "{"<<endl;

        f<<"\tint4 ind3d={get_global_id(0), get_global_id(1), get_global_id(2), 0};"<<endl;

        f<<"\tif(ind3d.x>=dim.x||ind3d.y>=dim.y||ind3d.z>=dim.z)"<<endl<<
         "\t\treturn;"<<endl;

        f<<"\tsize_t ind=d3To1d(ind3d, dim);"<<endl;

        for(size_t i=0; i<fieldNameAddTerms.size(); ++i){
            f<<"\tfloat field"<<i<<"=(g_newField+"<<"stride*"<<i<<")[ind];"<<endl;
        }

        //making a parameter list. It is the same for all reaction functions so far
        stringstream arg, argAdd;
        for(size_t i=0; i<fieldNameAddTerms.size(); ++i){
            arg<<"field"<<i;
            argAdd<<"field"<<i<<"+epsilon*(g_update+stride*"<<i<<")[ind]";
            if(i!=fieldNameAddTerms.size()-1){
                arg<<", ";
                argAdd<<", ";
            }
        }

        for(size_t i=0; i<fieldNameAddTerms.size(); ++i){
            f<<"\t(g_result+stride*"<<i<<")[ind]-="<<
             "dt*("<<funcNames[i]<<"("<<argAdd.str()<<")-"<<funcNames[i]<<"("<<arg.str()<<"))/epsilon;"<<endl;
        }

        f<<"}"<<endl;
    }

    return tmpKernelFileName;


}