#include <iostream>
#include <sstream>
#include "MiniLogger.h"

using namespace std;
using namespace CompuCell3D;

int main()
{

    // Log message C++ Interface
    MiniLogger* pLogger = NULL; // Create the object pointer for Logger Class
    pLogger = MiniLogger::getInstance();
    pLogger->initialize("my_log_file.txt", "file_log");

    //LoggerStream ls = pLogger->getLoggerStream("debug");

    //ls.log("This is a demo ", 2.0, 11, " str");

    (*pLogger) << LogMessageType::DEBUG_LOG << "this is log entry" << 12 << 14 << 15.1212;
    (*pLogger) << LogMessageType::TRACE_LOG << "this is log entry" << 12 << 14 << 15.1212;
    (*pLogger) << LogMessageType::ERROR_LOG << "this is log entry" << 12 << 14 << 15.1212;
    (*pLogger) << LogMessageType::INFO_LOG << "this is log entry" << 12 << 14 << 15.1212;


    //printDemo(12);


    //LOG_ALWAYS("<=============================== START OF PROGRAM ===============================>");
    //// Log message using Direct Interface
    //// Log Level: LOG_ERROR
    //LOG_ERROR("Message Logged using Direct Interface, Log level: LOG_ERROR");

    //LOG_ALARM("Message Logged using Direct Interface, Log level: LOG_ALARM");
    //LOG_ALWAYS("Message Logged using Direct Interface, Log level: LOG_ALWAYS");
    //LOG_INFO("Message Logged using Direct Interface, Log level: LOG_INFO");
    //LOG_BUFFER("Message Logged using Direct Interface, Log level: LOG_BUFFER");
    //LOG_TRACE("Message Logged using Direct Interface, Log level: LOG_TRACE");
    //LOG_DEBUG("Message Logged using Direct Interface, Log level: LOG_DEBUG");


    //pLogger->_error("Message Logged using C++ Interface, Log level: LOG_ERROR");
    //pLogger->_alarm("Message Logged using C++ Interface, Log level: LOG_ALARM");
    //pLogger->_always("Message Logged using C++ Interface, Log level: LOG_ALWAYS");
    //pLogger->_buffer("Message Logged using C++ Interface, Log level: LOG_INFO");
    //pLogger->_info("Message Logged using C++ Interface, Log level: LOG_BUFFER");
    //pLogger->_trace("Message Logged using C++ Interface, Log level: LOG_TRACE");
    //pLogger->_debug("Message Logged using C++ Interface, Log level: LOG_DEBUG");

    //// Log Variables
    //std::string name = "Pankaj Choudhary";
    //std::string address = "Delhi, India";
    //int age = 26;


    ////pLogger->enableConsoleLogging();
    //pLogger->updateLogLevel(INFO);


    //LOG_ALWAYS("<=============================== END OF PROGRAM ===============================>");
    //return 0;
}


//#include <iostream>
//#include <sstream>
//#include "Logger.h"
//
//using namespace std;
//using namespace CompuCell3D;
//
//int main()
//{
//
//   // Log message C++ Interface
//   Logger* pLogger = NULL; // Create the object pointer for Logger Class
//   pLogger = Logger::getInstance();
//   pLogger->initialize("my_log_file.txt", "file_log");
//
//   //LoggerStream ls = pLogger->getLoggerStream("debug");
//
//   //ls.log("This is a demo ", 2.0, 11, " str");
//
//   (*pLogger) << LogMessageType::DEBUG_LOG<< "this is log entry"<< 12<< 14<< 15.1212;
//   (*pLogger) << LogMessageType::TRACE_LOG << "this is log entry" << 12 << 14 << 15.1212;
//   (*pLogger) << LogMessageType::ERROR_LOG<< "this is log entry" << 12 << 14 << 15.1212;
//   (*pLogger) << LogMessageType::INFO_LOG << "this is log entry" << 12 << 14 << 15.1212;
//
//
//   //printDemo(12);
//
//
//   LOG_ALWAYS("<=============================== START OF PROGRAM ===============================>");
//   // Log message using Direct Interface
//   // Log Level: LOG_ERROR
//   LOG_ERROR("Message Logged using Direct Interface, Log level: LOG_ERROR");
//
//   LOG_ALARM("Message Logged using Direct Interface, Log level: LOG_ALARM");
//   LOG_ALWAYS("Message Logged using Direct Interface, Log level: LOG_ALWAYS");
//   LOG_INFO("Message Logged using Direct Interface, Log level: LOG_INFO");
//   LOG_BUFFER("Message Logged using Direct Interface, Log level: LOG_BUFFER");
//   LOG_TRACE("Message Logged using Direct Interface, Log level: LOG_TRACE");
//   LOG_DEBUG("Message Logged using Direct Interface, Log level: LOG_DEBUG");
//
//
//   pLogger->_error("Message Logged using C++ Interface, Log level: LOG_ERROR");
//   pLogger->_alarm("Message Logged using C++ Interface, Log level: LOG_ALARM");
//   pLogger->_always("Message Logged using C++ Interface, Log level: LOG_ALWAYS");
//   pLogger->_buffer("Message Logged using C++ Interface, Log level: LOG_INFO");
//   pLogger->_info("Message Logged using C++ Interface, Log level: LOG_BUFFER");
//   pLogger->_trace("Message Logged using C++ Interface, Log level: LOG_TRACE");
//   pLogger->_debug("Message Logged using C++ Interface, Log level: LOG_DEBUG");
//
//   // Log Variables
//   std::string name = "Pankaj Choudhary";
//   std::string address = "Delhi, India";
//   int age = 26;
//
//
//   //pLogger->enableConsoleLogging();
//   pLogger->updateLogLevel(INFO);
//
//   
//   LOG_ALWAYS("<=============================== END OF PROGRAM ===============================>");
//   return 0;
//}