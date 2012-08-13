#include <unitParserException.h>
#include <exception>
#include <stdexcept>
#include <string>

using namespace std;

extern "C" void throwParserException(char * _exceptionText){
	throw logic_error(_exceptionText);
	//throw logic_error("Scanning Error");


}
 
