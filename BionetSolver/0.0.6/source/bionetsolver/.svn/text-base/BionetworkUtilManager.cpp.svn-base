
#include "bionetsolver/BionetworkUtilManager.h"

#include <string>

BionetworkUtilManager::BionetworkUtilManager(){}
BionetworkUtilManager::~BionetworkUtilManager(){}

bool BionetworkUtilManager::charFoundInString(char find_char, std::string input_string){
    bool found = false;
    size_t pos = input_string.find(find_char);
    if(pos != -1) found = true;
    return found;
}

std::string BionetworkUtilManager::removeSpacesFromString(std::string input_string){
    std::string return_string = input_string;
    std::string::iterator sItr = return_string.begin();
    char space(' ');
    do{
        if(*sItr == space)
            sItr = return_string.erase(sItr);
        else
            ++sItr;
    }while(sItr != return_string.end());
    return return_string;
}

std::pair<std::string, std::string>
    BionetworkUtilManager::splitStringAtFirst(char split_char, std::string input_string){
    std::string::iterator split_location = input_string.begin();
    
    std::string lhs;
    std::string rhs;
    
    size_t pos = input_string.find(split_char);
    lhs = input_string.substr(0,pos);
    rhs = input_string.substr(pos+1);
    
    return std::pair<std::string, std::string>(lhs, rhs);
}







