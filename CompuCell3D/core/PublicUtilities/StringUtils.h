#ifndef STRINGUTILS_H
#define STRINGUTILS_H

#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <locale>
#include <cctype>
#include <functional>
#include <iostream>

// ----------------------------
// Function declarations
// ----------------------------

void parseStringIntoList(std::string &str, std::vector<std::string> &strVec, std::string separator);

/**
 * @brief Splits a string into a vector of strings by a given token
 *
 * @param str string to split
 * @param token token by which to split the string
 * @return vector of strings
 */
std::vector<std::string> splitString(const std::string& str, const std::string& token);


// ----------------------------
// Functors (C++17 compliant)
// ----------------------------

class isWhiteSpaceFunctor {
private:
    std::locale loc;

public:
    isWhiteSpaceFunctor() : loc(std::locale()) {}

    bool operator()(char c) const {
        // C++17: std::isspace(c) is safe since it uses global C locale
        return std::isspace(static_cast<unsigned char>(c));
    }
};


struct ToLower
{
    explicit ToLower(const std::locale& l = std::locale()) : loc(l) {}

    char operator()(char c) const {
        return static_cast<char>(std::tolower(static_cast<unsigned char>(c), loc));
    }

private:
    const std::locale& loc;
};


struct ToUpper
{
    explicit ToUpper(const std::locale& l = std::locale()) : loc(l) {}

    char operator()(char c) const {
        return static_cast<char>(std::toupper(static_cast<unsigned char>(c), loc));
    }

private:
    const std::locale& loc;
};


// ----------------------------
// Conversion helpers
// ----------------------------

void changeToLower(std::string & str);
void changeToUpper(std::string & str);
std::string strToUpper(const std::string &str);
std::string strToLower(const std::string &str);
bool strToBool(const std::string &str);
int strToInt(const std::string &str);
unsigned int strToUInt(const std::string &str);
short strToShort(const std::string &str);
unsigned short strToUShort(const std::string &str);
double strToDouble(const std::string &str);
char strToByte(const std::string &str);
unsigned char strToUByte(const std::string &str);

#endif

//#ifndef STRINGUTILS_H
//#define STRINGUTILS_H
//#include <vector>
//#include <cstring>
//#include <algorithm>
//#include <locale>
//#include <cctype>
//#include <functional>
//#include <iostream>
//
//
//
//void parseStringIntoList(std::string &str,std::vector<std::string> &strVec,std::string separator);
///**
// * @brief Splits a string into a vector of strings by a given token
// *
// * @param str string to split
// * @param token token by which to split the string
// * @return vector of strings
// */
//std::vector<std::string> splitString(const std::string& str, const std::string& token);
//
//class isWhiteSpaceFunctor: public std::unary_function<char,bool>{
//   private:
//   std::locale loc;
//
//   public:
//      bool operator()(char c)const{
//
////         return std::isspace(c,loc);
//          return std::isspace(c);
//
//      }
//
//};
//
//
//struct ToLower
//{
//   ToLower(std::locale const& l) : loc(l) {;}
//   char operator() (char c) const  {
////       return std::tolower(c,loc);
//       return std::tolower(c);
//   }
//   private:
//      std::locale const& loc;
//};
//
//
//struct ToUpper
//{
//      ToUpper(std::locale const& l) : loc(l) {;}
//      char operator() (char c) const  {
////          return std::toupper(c,loc);
//          return std::toupper(c);
//      }
//private:
//      std::locale const& loc;
//};
//
//void changeToLower(std::string & str);
//void changeToUpper(std::string & str);
//std::string strToUpper(const std::string &str);
//std::string strToLower(const std::string &str);
//bool strToBool(const std::string &str);
//int strToInt(const std::string &str);
//unsigned int strToUInt(const std::string &str);
//short strToShort(const std::string &str);
//unsigned short strToUShort(const std::string &str);
//double strToDouble(const std::string &str);
//char strToByte(const std::string &str);
//unsigned char strToUByte(const std::string &str);
//
//#endif
