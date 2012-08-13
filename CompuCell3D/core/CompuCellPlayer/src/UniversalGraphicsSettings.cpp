#include <UniversalGraphicsSettings.h>

bool UniversalGraphicsSettings::avoidType(unsigned short type){
   

   for(unsigned int i = 0 ; i < types3DInvisibleVec.size() ; ++i){
/*      cerr<<"i="<<i;
      cerr<<" type="<<type<<" avoid="<<univGraphSetPtr->types3DInvisibleVec[i]<<endl;*/
      if(type==types3DInvisibleVec[i]){
//          exit(0);
         return true;
      }
   }
//    exit(0);
   return false;   

}

