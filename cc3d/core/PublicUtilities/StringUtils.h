#ifndef STRINGUTILS_H
#define STRINGUTILS_H
#include <vector>
#include <string>
#include <algorithm>
#include <locale>
#include <cctype>
#include <functional>

void parseStringIntoList(std::string &str,std::vector<std::string> &strVec,std::string separator);

class isWhiteSpaceFunctor: public std::unary_function<char,bool>{
   private:
   std::locale loc;
   
   public:
      bool operator()(char c)const{
         return isspace(c,loc);

      }

};


struct ToLower
{
   ToLower(std::locale const& l) : loc(l) {;}
   char operator() (char c) const  { return std::tolower(c,loc); }
   private:
      std::locale const& loc;
};


struct ToUpper
{
      ToUpper(std::locale const& l) : loc(l) {;}
      char operator() (char c) const  { return std::toupper(c,loc); }
private:
      std::locale const& loc;
};

void changeToLower(std::string & str);
void changeToUpper(std::string & str);
#endif
