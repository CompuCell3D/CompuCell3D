#include <stdlib.h>
#include <iostream>
#include <ostream>
#include <chrono>
#include <time.h>
#include "..\core\BasicUtils\BasicRandomNumberGenerator.h"

using namespace std;


long runRandomNumberBenchmark(int latticeDim, int numSteps) {

	cout << "Timing generation of random integers for a " << latticeDim << "x" << latticeDim << " lattice over " << numSteps << " MCSs." << endl;

	BasicRandomNumberGenerator *randGen = BasicRandomNumberGenerator::getInstance();

	srand(time(0));
	unsigned int randomSeed = (unsigned int)rand()*((std::numeric_limits<unsigned int>::max)() - 1);
	randGen->setSeed(randomSeed);

	long ptX, ptY;

	auto startTime = chrono::high_resolution_clock::now();

	for (unsigned int x = 0; x < latticeDim; ++x)
		for (unsigned int y = 0; y < latticeDim; ++y)
			for (unsigned int MCS = 0; MCS < numSteps; ++MCS) {
				ptX = randGen->getInteger(0, latticeDim - 1);
				ptY = randGen->getInteger(0, latticeDim - 1);
			}

	auto endTime = chrono::high_resolution_clock::now();

	auto execTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);

	return execTime.count();

}

int main(){
	int latticeDim = 100;
	int numSteps = 1000;

	auto t = runRandomNumberBenchmark(latticeDim, numSteps);
	cout << t << " microseconds" << endl;
	
	return 0;
}
