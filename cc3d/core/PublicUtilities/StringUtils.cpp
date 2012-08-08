#include "StringUtils.h"
#include <iostream>
#include <algorithm>

using namespace std;
void parseStringIntoList(std::string &str,std::vector<std::string> &strVec,std::string separator){

   string::size_type beginPos;
   string::size_type endPos;
   string::size_type strSize;
   string::size_type badPos = string::npos;
   std::locale loc;
   
   if(str.empty())
      return;
      

   //removing spaces from string
/*   str.erase   (
               remove_if(str.begin(), str.end(), bind2nd(isspace(),loc) ),
   
               str.end()
               );*/
   str.erase   (
               remove_if(str.begin(), str.end(),isWhiteSpaceFunctor()
               
                 ),
   
               str.end()
               );

                     
   strSize=str.size();
   endPos=0;
   for(;;){
      beginPos=endPos;
      endPos=str.find(separator,beginPos);
      if(endPos==badPos)
         break;

      strVec.push_back(string().assign(str,beginPos,endPos-beginPos));
/*      cerr<<*(--strVec.end())<<endl;
      cerr<<string().assign(str,endPos,strSize-endPos)<<endl;*/
      endPos+=separator.size();
      //exit(0);
   }

   strVec.push_back(string().assign(str,beginPos,strSize-endPos));
   // have to reverse so that ordering is the same as in the parsed string
   reverse(strVec.begin(),strVec.end());
   
}

void changeToLower(std::string & str){

   std::locale loc("C");
   ToLower low(loc);
   transform(str.begin(),str.end(),str.begin(),low);

}

void changeToUpper(std::string & str){

   std::locale loc("C");
   ToUpper up(loc);
   transform(str.begin(),str.end(),str.begin(),up);

}
