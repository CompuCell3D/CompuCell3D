#pragma hdrstop
#pragma argsused

#include <iostream>
#include "Poco/UUIDGenerator.h"
#include "Poco/UUIDGenerator.h"

using Poco::UUID;
using Poco::UUIDGenerator;

int main()
{
	UUIDGenerator& generator = UUIDGenerator::defaultGenerator();
	UUID uuid2(generator.createRandom());
	UUID uuid3(generator.createRandom());
	std::cout << uuid2.toString() << std::endl;
	std::cout << uuid3.toString() << std::endl;
    return 0;
}


#pragma comment(lib, "poco_foundation-static.lib")