#include <iostream>
#include <sstream>
#include "Logger.h"

using namespace std;
using namespace CompuCell3D;


int main()
{

    // Log message C++ Interface
    Logger* pLogger = NULL; // Create the object pointer for Logger Class
    pLogger = Logger::getInstance();
    pLogger->initialize("my_log_file.txt", "file_log");


    (*pLogger) << LogMessageType::DEBUG_LOG << "this is log entry" << 12 << 14 << 15.1212;
    (*pLogger) << LogMessageType::TRACE_LOG << "this is log entry" << 12 << 14 << 15.1212;
    (*pLogger) << LogMessageType::ERROR_LOG << "this is log entry" << 12 << 14 << 15.1212;
    (*pLogger) << LogMessageType::INFO_LOG << "this is log entry" << 12 << 14 << 15.1212;

    return 0;
}