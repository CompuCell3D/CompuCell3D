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

   //LoggerStream ls = pLogger->getLoggerStream("debug");

   //ls.log("This is a demo ", 2.0, 11, " str");

   pLogger->error("error This is a demo ", 2.0, 11, " str");
   (*pLogger) << DEBUG<< "this is log entry"<< 12<< 14<< 15.1212;
   //printDemo(12);


   LOG_ALWAYS("<=============================== START OF PROGRAM ===============================>");
   // Log message using Direct Interface
   // Log Level: LOG_ERROR
   LOG_ERROR("Message Logged using Direct Interface, Log level: LOG_ERROR");
   LOG_ERROR(to_str("Fount error with value", 10.0, "and", 2));

   LOG_ALARM("Message Logged using Direct Interface, Log level: LOG_ALARM");
   LOG_ALWAYS("Message Logged using Direct Interface, Log level: LOG_ALWAYS");
   LOG_INFO("Message Logged using Direct Interface, Log level: LOG_INFO");
   LOG_BUFFER("Message Logged using Direct Interface, Log level: LOG_BUFFER");
   LOG_TRACE("Message Logged using Direct Interface, Log level: LOG_TRACE");
   LOG_DEBUG("Message Logged using Direct Interface, Log level: LOG_DEBUG");


   pLogger->_error("Message Logged using C++ Interface, Log level: LOG_ERROR");
   pLogger->_alarm("Message Logged using C++ Interface, Log level: LOG_ALARM");
   pLogger->_always("Message Logged using C++ Interface, Log level: LOG_ALWAYS");
   pLogger->_buffer("Message Logged using C++ Interface, Log level: LOG_INFO");
   pLogger->_info("Message Logged using C++ Interface, Log level: LOG_BUFFER");
   pLogger->_trace("Message Logged using C++ Interface, Log level: LOG_TRACE");
   pLogger->_debug("Message Logged using C++ Interface, Log level: LOG_DEBUG");

   // Log Variables
   std::string name = "Pankaj Choudhary";
   std::string address = "Delhi, India";
   int age = 26;

   std::ostringstream ss;
   ss << endl;
   ss << "\t" << "My Contact Details:" << endl;
   ss << "\t" << "Name: " << name << endl;
   ss << "\t" << "Address: " << address << endl;
   ss << "\t" << "Age: " << age << endl << endl;

   //pLogger->enableConsoleLogging();
   pLogger->updateLogLevel(INFO);

   // Log ostringstream ss to all the log levels
   LOG_ALWAYS("Logging ostringstream using Direct Interface");
   LOG_ERROR(ss);
   LOG_ALARM(ss);
   LOG_ALWAYS(ss);
   LOG_INFO(ss);
   LOG_BUFFER(ss);
   LOG_TRACE(ss);
   LOG_DEBUG(ss);

   Logger::getInstance()->_buffer("Logging ostringstream using C++ Interface");
   Logger::getInstance()->_error(ss);
   Logger::getInstance()->_alarm(ss);
   Logger::getInstance()->_always(ss);
   Logger::getInstance()->_buffer(ss);
   Logger::getInstance()->_info(ss);
   Logger::getInstance()->_trace(ss);
   Logger::getInstance()->_debug(ss);
   
   LOG_ALWAYS("<=============================== END OF PROGRAM ===============================>");
   return 0;
}