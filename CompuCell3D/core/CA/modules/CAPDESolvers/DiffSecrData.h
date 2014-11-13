#ifndef DIFFSECRDATA_H
#define DIFFSECRDATA_H


#include <string>

#include <vector>
#include <set>
#include <map>
#include <iostream>
#include <limits>
#include <climits>
#undef max
#undef min



#include "CAPDESolversDLLSpecifier.h"


namespace CompuCell3D {


class CAPDESOLVERS_EXPORT DiffusionData  {
      
   public:
      DiffusionData():
      
      diffConst(0.0),
      decayConst(0.0),
      timesPerMCS(0)
	  {}
      float diffConst;
      float decayConst; 
      unsigned int timesPerMCS;
      std::string name;
      
      friend std::ostream & operator<<(std::ostream & out,CompuCell3D::DiffusionData & diffData);
};


class CAPDESOLVERS_EXPORT SecretionData{
   protected:
      
   public:
      SecretionData()
      {
		  secretionConst.assign(UCHAR_MAX+1,0.0);
                      
      }
	  float getSecrConst(unsigned int i){return secretionConst[i];}
	  void setSecrConst(unsigned int i,float val){secretionConst[i]=val;}
      void setTypeNameSecrConst(std::string typeName,float val){secretionMap[typeName]=val;}

      std::string name;

      std::vector<float>  secretionConst;
      std::map<std::string, float> secretionMap;
	  


   
};



inline std::ostream & operator<<(std::ostream & out,CompuCell3D::DiffusionData & diffData){
   using namespace std;
   
   out<<"DiffusionConstant: "<<diffData.diffConst<<endl;
   out<<"DecayConstant: "<<diffData.decayConst<<endl;

   return out;
}




};

#endif
