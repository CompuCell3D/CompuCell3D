#include "FileUtils.h"
#include <fstream>
#include <string>
using namespace std;

bool copyFile(const char *inName , const char * outName){
      
      
      ifstream in(inName);
      ofstream out(outName);
      char c;  
      while(in.good()){
         in.get(c);
         
         if(!in.eof()){
            out.put(c);
         }else{
            return false;
         }
      }
      return true;
}