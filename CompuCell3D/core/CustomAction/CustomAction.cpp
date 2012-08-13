	// CustomAction.cpp : Defines the entry point for the console application.
//


#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

#include <direct.h> // for getcwd

#include <stdlib.h>// for MAX_PATH
#include "FileUtils.h"

using namespace std;
int main(int argc, char* argv[])
{
	char *instDirCharPtr=argv[1];

	string instDir(instDirCharPtr);
	instDir.erase(instDir.size()-1,1);

	chdir(instDir.c_str());
	system("python scriptSetup.py");

	return 0;
}

