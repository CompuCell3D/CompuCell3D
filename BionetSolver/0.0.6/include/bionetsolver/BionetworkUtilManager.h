

#ifndef BIONETWORKUTILMANAGER_H
#define BIONETWORKUTILMANAGER_H


#include <iostream>
#include <map>
#include <vector>

#include <math.h>
#include "BionetworkDLLSpecifier.h"

class BIONETWORK_EXPORT BionetworkUtilManager{
    
    public:
    
        BionetworkUtilManager();
        ~BionetworkUtilManager();
        
        // String utilities
        std::string removeSpacesFromString(std::string);
        std::pair<std::string, std::string> splitStringAtFirst(char, std::string);
        bool charFoundInString(char, std::string);
        
};


#endif