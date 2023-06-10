#include "StringUtils.h"
#include <iostream>
#include <algorithm>
#include <string>

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
      endPos+=separator.size();
      //exit(0);
   }

   strVec.push_back(string().assign(str,beginPos,strSize-endPos));
   // have to reverse so that ordering is the same as in the parsed string
   reverse(strVec.begin(),strVec.end());
   
}

std::vector<std::string> splitString(const std::string& str, const std::string& token) {
   std::vector<std::string> o;
   char* cStr = strdup(str.c_str());
   char* cTok = strdup(token.c_str());
   char* oStr = strtok(cStr, cTok);
   while (oStr) {
      o.push_back(oStr);
      oStr = strtok(NULL, cTok);
   }
   return o;
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

string toUpper(const string &str) {
   string s(str);
   changeToUpper(s);
   return s;
}

string toLower(const string &str) {
   string s(str);
   changeToLower(s);
   return s;
}

bool strToBool(const string &str) {
   string s = toLower(str);
   if (s == "true") return true;
   if (s == "false") return false;
   throw string("Invalid bool: ") + str;
}

int strToInt(const std::string &str) {
	return stoi(str);
}

unsigned int strToUInt(const std::string &str) {
	return (unsigned int)strToInt(str);
}

short strToShort(const std::string &str) {
	return (short)strToInt(str);
}

unsigned short strToUShort(const std::string &str) {
	return (unsigned short)strToInt(str);
}

double strToDouble(const std::string &str) {
	return stod(str);
}

char strToByte(const std::string &str) {
	return (char)strToUInt(str);
}

unsigned char strToUByte(const std::string &str) {
	return (unsigned char)strToUInt(str);
}
