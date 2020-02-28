#include <stdlib.h>
#include <iostream>
#include <ostream>
#include <chrono>
#include <time.h>

#include "..\core\BasicUtils\BasicRandomNumberGenerator.h"
#include <random>

using namespace std;


long runBasicRandomNumberGeneratorBenchmark(int latticeDim, int numSteps) {

	srand(time(0));
	unsigned int randomSeed = (unsigned int)rand()*((std::numeric_limits<unsigned int>::max)() - 1);

	BasicRandomNumberGenerator *randGen = BasicRandomNumberGenerator::getInstance();
	randGen->setSeed(randomSeed);

	long ptX, ptY;

	unsigned int numIts = latticeDim*latticeDim*numSteps;

	auto startTime = chrono::high_resolution_clock::now();

	for (unsigned int it = 0; it < numIts; ++it) {
		ptX = randGen->getInteger(0, latticeDim - 1);
		ptY = randGen->getInteger(0, latticeDim - 1);
	}

	auto endTime = chrono::high_resolution_clock::now();

	auto execTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);

	return execTime.count();

}

long runBasicRandomNumberGeneratorIndBenchmark(int latticeDim, int numSteps) {

	srand(time(0));
	unsigned int randomSeed = (unsigned int)rand()*((std::numeric_limits<unsigned int>::max)() - 1);

	BasicRandomNumberGenerator *randGen = BasicRandomNumberGenerator::getInstance();
	randGen->setSeed(randomSeed);

	long ind, ptX, ptY;

	unsigned int maxInd = latticeDim*latticeDim - 1;
	unsigned int numIts = latticeDim*latticeDim*numSteps;

	auto startTime = chrono::high_resolution_clock::now();

	for (unsigned int it = 0; it < numIts; ++it) {
		
		ind = randGen->getInteger(0, maxInd);
		ptX = ind % latticeDim;
		ptY = (ind - ptX) / latticeDim % latticeDim;
	}

	auto endTime = chrono::high_resolution_clock::now();

	auto execTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);

	return execTime.count();

}

long runRandBenchmark(int latticeDim, int numSteps) {

	srand(time(0));
	long randomSeed = (long)rand()*((std::numeric_limits<long>::max)() - 1);

	long ptX, ptY;

	unsigned int numIts = latticeDim*latticeDim*numSteps;

	auto startTime = chrono::high_resolution_clock::now();

	for (unsigned int it = 0; it < numIts; ++it) {
		ptX = std::rand() % latticeDim;
		ptY = std::rand() % latticeDim;
	}

	auto endTime = chrono::high_resolution_clock::now();

	auto execTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);

	return execTime.count();

}

long runMersenneTwisterBenchmark(int latticeDim, int numSteps) {

	srand(time(0));
	long randomSeed = (long)rand()*((std::numeric_limits<long>::max)() - 1);

	mt19937 randGen(randomSeed);

	long ptX, ptY;

	unsigned int numIts = latticeDim*latticeDim*numSteps;

	auto startTime = chrono::high_resolution_clock::now();

	for (unsigned int it = 0; it < numIts; ++it) {
		ptX = uniform_int_distribution<long>(0, latticeDim - 1)(randGen);
		ptY = uniform_int_distribution<long>(0, latticeDim - 1)(randGen);
	}

	auto endTime = chrono::high_resolution_clock::now();

	auto execTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);

	return execTime.count();

}

long runMersenneTwisterBenchmark64(int latticeDim, int numSteps) {

	srand(time(0));
	long randomSeed = (long)rand()*((std::numeric_limits<long>::max)() - 1);

	mt19937_64 randGen(randomSeed);

	long ptX, ptY;

	unsigned int numIts = latticeDim*latticeDim*numSteps;

	auto startTime = chrono::high_resolution_clock::now();

	for (unsigned int it = 0; it < numIts; ++it) {
		ptX = uniform_int_distribution<long>(0, latticeDim - 1)(randGen);
		ptY = uniform_int_distribution<long>(0, latticeDim - 1)(randGen);
	}

	auto endTime = chrono::high_resolution_clock::now();

	auto execTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);

	return execTime.count();

}

int main(){
	int latticeDim = 100;
	int numSteps = 1000;

	float numCalls = float(latticeDim*latticeDim*numSteps);

	cout << endl;

	cout << "Timing generation of random integers for a " << latticeDim << "x" << latticeDim << " lattice over " << numSteps << " MCSs." << endl;

	auto tBasicRandomNumberGenerator = runBasicRandomNumberGeneratorBenchmark(latticeDim, numSteps);
	cout << "BasicRandomNumberGenerator   : " << tBasicRandomNumberGenerator << " microseconds" << endl;
	cout << "                             : " << 1000.0 * float(tBasicRandomNumberGenerator) / numCalls << " nanoseconds/call" << endl;

	auto tBasicRandomNumberGeneratorInd = runBasicRandomNumberGeneratorIndBenchmark(latticeDim, numSteps);
	cout << "BasicRandomNumberGeneratorInd: " << tBasicRandomNumberGeneratorInd << " microseconds" << endl;
	cout << "                             : " << 1000.0 * float(tBasicRandomNumberGeneratorInd) / numCalls << " nanoseconds/call" << endl;

	auto tRandBenchmark = runRandBenchmark(latticeDim, numSteps);
	cout << "rand                         : " << tRandBenchmark << " microseconds" << endl;
	cout << "                             : " << 1000.0 * float(tRandBenchmark) / numCalls << " nanoseconds/call" << endl;

	auto tMersenneTwisterBenchmark = runMersenneTwisterBenchmark(latticeDim, numSteps);
	cout << "mt19937                      : " << tMersenneTwisterBenchmark << " microseconds" << endl;
	cout << "                             : " << 1000.0 * float(tMersenneTwisterBenchmark) / numCalls << " nanoseconds/call" << endl;

	auto tMersenneTwisterBenchmark64 = runMersenneTwisterBenchmark64(latticeDim, numSteps);
	cout << "mt19937_64                   : " << tMersenneTwisterBenchmark64 << " microseconds" << endl;
	cout << "                             : " << 1000.0 * float(tMersenneTwisterBenchmark64) / numCalls << " nanoseconds/call" << endl;
	
	return 0;
}
