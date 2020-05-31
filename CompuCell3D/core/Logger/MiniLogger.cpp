///////////////////////////////////////////////////////////////////////////////
// @File Name:     Logger.cpp                                                //
// @Author:        Pankaj Choudhary                                          //
// @Version:       0.0.1                                                     //
// @L.M.D:         13th April 2015                                           //
// @Description:   For Logging into file                                     //
//                                                                           // 
// Detail Description:                                                       //
// Implemented complete logging mechanism, Supporting multiple logging type  //
// like as file based logging, console base logging etc. It also supported   //
// for different log type.                                                   //
//                                                                           //
// Thread Safe logging mechanism. Compatible with VC++ (Windows platform)   //
// as well as G++ (Linux platform)                                           //
//                                                                           //
// Supported Log Type: ERROR, ALARM, ALWAYS, INFO, BUFFER, TRACE, DEBUG      //
//                                                                           //
// No control for ERROR, ALRAM and ALWAYS messages. These type of messages   //
// should be always captured.                                                //
//                                                                           //
// BUFFER log type should be use while logging raw buffer or raw messages    //
//                                                                           //
// Having direct interface as well as C++ Singleton inface. can use          //
// whatever interface want.                                                   //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

// C++ Header File(s)
#include <iostream>
#include <cstdlib>
#include <ctime>

#include "MiniLogger.h"

using namespace std;
using namespace CompuCell3D;


/* |IMPLEMENTATION| User.cpp file */

#include "MiniLogger.h" 
#include <iostream> 
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <unordered_map>
#include <mutex>
#include <functional> 

using namespace std;

MiniLogger* MiniLogger::m_Instance = 0;

MiniLogger* MiniLogger::getInstance() throw ()
{
    if (m_Instance == 0)
    {
        m_Instance = new MiniLogger();
    }
    return m_Instance;
}

class MiniLogger::Impl {
private:

    std::ofstream           m_File;
    
    std::mutex m_Mutex;
    
    std::string _log_type;
    std::string _log_level;
    std::string _log_fname;
    
    LogLevel                m_LogLevel;
    LogType                 m_LogType;
    
    const std::map<std::string, LogType> log_type_map = {
        { "no_log", LogType::NO_LOG },
        { "console_log", LogType::CONSOLE_LOG },
        { "file_log", LogType::FILE_LOG }
    };
    
    const std::map<std::string, LogLevel> log_level_map = {
        { "disable_log",  LogLevel::NO_LOG_LEVEL },
        { "log_level_info", LogLevel::INFO },
        { "log_level_buffer", LogLevel::BUFFER },
        { "log_level_trace", LogLevel::TRACE },
        { "enable_log", LogLevel::ALL_LOG },
        {}
    };
    
    LogType stringToLogType(std::string log_type_str) {
        std::map<std::string, LogType>::const_iterator mitr = log_type_map.find(log_type_str);
        if (mitr != log_type_map.end()) {
            return mitr->second;
        }
        else {
            return LogType::CONSOLE_LOG;
        }
    }
    LogLevel stringToLogLevel(std::string log_level_str) {
        std::map<std::string, LogLevel>::const_iterator mitr = log_level_map.find(log_level_str);
        if (mitr != log_level_map.end()) {
            return mitr->second;
        }
        else {
            return LogLevel::ALL_LOG;
        }
    }

    void lock(){
        m_Mutex.lock();
    }
    
    void unlock() {
        m_Mutex.unlock();
    }
    
    std::string getCurrentTime() {
    
        string currTime;
        //Current date/time based on current time
        time_t now = time(0);
        // Convert current time to string
        currTime.assign(ctime(&now));
    
        // Last charactor of currentTime is "\n", so remove it
        string currentTime = currTime.substr(0, currTime.size() - 1);
        return currentTime;
    }
    
    void logIntoFile(std::string& data) {
        lock();
        m_File << getCurrentTime() << "  " << data << endl;
        m_File.flush();
        unlock();
    
    }
    void logOnConsole(std::string& data) {
        cout << getCurrentTime() << "  " << data << endl;
    }


public:
    Impl(){};

    ~Impl() {
        m_File.close();
    }

    void initialize(std::string log_fname, std::string log_type = "file_log", std::string log_level = "enable_log") {
        _log_level = log_level;
        _log_type = log_type;
        _log_fname = log_fname;
        m_LogType = stringToLogType(log_type);
        m_LogLevel = stringToLogLevel(log_level);
        
        
        if (log_fname.size()) {
        
            if (m_LogType == LogType::FILE_LOG) {
                m_File.open(log_fname, ios::out | ios::app);
            }
            else {
                m_LogType = LogType::CONSOLE_LOG;
            }
        }

    }

    void _error(const char* text) throw() {
        string data;
        data.append("[ERROR]: ");
        data.append(text);

        // ERROR must be capture
        if (m_LogType == LogType::FILE_LOG)
        {
            logIntoFile(data);
        }
        else if (m_LogType == LogType::CONSOLE_LOG)
        {
            logOnConsole(data);
        }

    }

    // Interface for Alarm Log 
    void _alarm(const char* text) throw() {
        string data;
        data.append("[ALARM]: ");
        data.append(text);

        // ALARM must be capture
        if (m_LogType == LogType::FILE_LOG)
        {
            logIntoFile(data);
        }
        else if (m_LogType == LogType::CONSOLE_LOG)
        {
            logOnConsole(data);
        }

    }

    // Interface for Always Log 
    void _always(const char* text) throw() {
        string data;
        data.append("[ALWAYS]: ");
        data.append(text);

        // No check for ALWAYS logs
        if (m_LogType == LogType::FILE_LOG)
        {
            logIntoFile(data);
        }
        else if (m_LogType == LogType::CONSOLE_LOG)
        {
            logOnConsole(data);
        }
    }

    // Interface for Buffer Log 
    void _buffer(const char* text) throw() {
        // Buffer is the special case. So don't add log level
        // and timestamp iLogType:n the buffer message. Just log the raw bytes.
        if ((m_LogType == LogType::FILE_LOG) && (m_LogLevel >= LogLevel::BUFFER))
        {
            lock();
            m_File << text << endl;
            unlock();
        }
        else if ((m_LogType == LogType::CONSOLE_LOG) && (m_LogLevel >= LogLevel::BUFFER))
        {
            cout << text << endl;
        }
    }

    // Interface for Info Log 
    void _info(const char* text) throw() {
        string data;
        data.append("[INFO]: ");
        data.append(text);

        if ((m_LogType == LogType::FILE_LOG) && (m_LogLevel >= LogLevel::INFO))
        {
            logIntoFile(data);
        }
        else if ((m_LogType == LogType::CONSOLE_LOG) && (m_LogLevel >= LogLevel::INFO))
        {
            logOnConsole(data);
        }
    }

    // Interface for Trace log 
    void _trace(const char* text) throw() {
        string data;
        data.append("[TRACE]: ");
        data.append(text);

        if ((m_LogType == LogType::FILE_LOG) && (m_LogLevel >= LogLevel::TRACE))
        {
            logIntoFile(data);
        }
        else if ((m_LogType == LogType::CONSOLE_LOG) && (m_LogLevel >= LogLevel::TRACE))
        {
            logOnConsole(data);
        }
    }


    void _debug(const char* text) {
        string data;
        data.append("[DEBUG]: ");
        data.append(text);
        
        if ((m_LogType == LogType::FILE_LOG) && (m_LogLevel >= LogLevel::DEBUG))
        {
            logIntoFile(data);
        }
        else if ((m_LogType == LogType::CONSOLE_LOG) && (m_LogLevel >= LogLevel::DEBUG))
        {
            logOnConsole(data);
        }
    }    

    void updateLogLevel(LogLevel logLevel) {
        m_LogLevel = logLevel;
    }
    
    // Enable all log levels
    void enaleLog()
    {
        m_LogLevel = LogLevel::ALL_LOG;
    }
    
    // Disable all log levels, except error and alarm
    void disableLog()
    {
        m_LogLevel = LogLevel:: NO_LOG_LEVEL;
    }
    
    // Interfaces to control log Types
    void updateLogType(LogType logType)
    {
        m_LogType = logType;
    }
    
    void enableConsoleLogging()
    {
        m_LogType = LogType::CONSOLE_LOG;
    }
    
    void enableFileLogging()
    {
        m_LogType = LogType::FILE_LOG;
    }

    std::string getLogType() {
        return _log_type;
    }

    std::string getLogLevel() {
        return _log_level;
    }

    std::string getLogFname() {
        return _log_fname;
    }

    void welcomeMessage()
    {
        cout << "Welcome "<< name << endl;
    }

    string name;
    int salary = -1;
};



/// <summary>
/// //////////////////////////
/// </summary>



// Constructor connected with our Impl structure 
MiniLogger::MiniLogger()
    : pimpl(std::make_unique<Impl>())
{
    pimpl->welcomeMessage();
}

void MiniLogger::initialize(std::string log_fname, std::string log_type, std::string log_level) {
    this->pimpl->initialize(log_fname, log_type, log_level);
}

// Default Constructor 
MiniLogger::~MiniLogger() = default;

std::string MiniLogger::getLogType() {
    return this->pimpl->getLogType();    
}

std::string MiniLogger::getLogLevel() {
    return this->pimpl->getLogLevel();
}

std::string MiniLogger::getLogFname() {
    return this->pimpl->getLogFname();
}

// Interfaces to control log levels
void MiniLogger::updateLogLevel(LogLevel logLevel)
{
    this->pimpl->updateLogLevel(logLevel);   
}

// Enable all log levels
void MiniLogger::enaleLog()
{
    this->pimpl->enaleLog();    
}

// Disable all log levels, except error and alarm
void MiniLogger::disableLog()
{
    this->pimpl->disableLog();    
}

// Interfaces to control log Types
void MiniLogger::updateLogType(LogType logType)
{
    this->pimpl->updateLogType(logType);
}

void MiniLogger::enableConsoleLogging()
{
    this->pimpl->enableConsoleLogging();
}

void MiniLogger::enableFileLogging()
{
    this->pimpl->enableFileLogging();
}

void  MiniLogger::_debug(const char* text) {
    this->pimpl->_debug(text);
}

void MiniLogger::_error(const char* text) throw(){
    this->pimpl->_error(text);
}

// Interface for Alarm Log 
void MiniLogger::_alarm(const char* text) throw() {
    this->pimpl->_alarm(text);
}
// Interface for Always Log 
void MiniLogger::_always(const char* text) throw() {
    this->pimpl->_always(text);
}
// Interface for Buffer Log 
void MiniLogger::_buffer(const char* text) throw() {
    this->pimpl->_buffer(text);
}
// Interface for Info Log 
void MiniLogger::_info(const char* text) throw() {
    this->pimpl->_info(text);
}

// Interface for Trace log 
void MiniLogger::_trace(const char* text) throw() {
    this->pimpl->_trace(text);
}

// Assignment operator and Copy constructor 

//MiniLogger::MiniLogger(const MiniLogger& other)
//    : pimpl(new Impl(*other.pimpl))
//{
//}
//
//MiniLogger& MiniLogger::operator=(MiniLogger rhs)
//{
//    swap(pimpl, rhs.pimpl);
//    return *this;
//}

// Getter and setter 
int MiniLogger::getSalary()
{
    return pimpl->salary;
}

void MiniLogger::setSalary(int salary)
{
    pimpl->salary = salary;
    cout << "Salary set to "
        << salary << endl;
}





template<>
MiniLoggerStream CompuCell3D::operator<<(MiniLogger& logger, const LogMessageType &  val) {
    MiniLoggerStream logger_stream(&logger);
    logger_stream.setLogLevel(val);
    return logger_stream;
}



//MiniLogger* MiniLogger::m_Instance = 0;
//
//
//struct MiniLogger::MiniLoggerImpl {
//
//
//    std::ofstream           m_File;
//
//    std::mutex m_Mutex;
//
//    std::string _log_type;
//    std::string _log_level;
//    std::string _log_fname;
//
//    LogLevel                m_LogLevel;
//    LogType                 m_LogType;
//
//    const std::map<std::string, LogType> log_type_map = {
//        { "no_log", NO_LOG },
//        { "console_log", CONSOLE_LOG },
//        { "file_log", FILE_LOG }
//    };
//
//    const std::map<std::string, LogLevel> log_level_map = {
//        { "disable_log", NO_LOG_LEVEL },
//        { "log_level_info", INFO },
//        { "log_level_buffer", BUFFER },
//        { "log_level_trace", TRACE },
//        { "enable_log", ALL_LOG },
//        {}
//    };
//
//
//    //void initialize(std::string fname=0, LogType log_type = CONSOLE_LOG);
//    ~MiniLoggerImpl() {
//        m_File.close();
//    }
//
//    void initialize(std::string log_fname, std::string log_type = "file_log", std::string log_level = "enable_log") {
//
//        _log_level = log_level;
//        _log_type = log_type;
//        _log_fname = log_fname;
//        m_LogType = stringToLogType(log_type);
//        m_LogLevel = stringToLogLevel(log_level);
//
//
//        if (log_fname.size()) {
//
//            if (m_LogType == FILE_LOG) {
//                m_File.open(log_fname, ios::out | ios::app);
//            }
//            else {
//                m_LogType = CONSOLE_LOG;
//            }
//        }
//
//
//    }
//    // Interface for Error Log 
//
//    // Error and Alarm log must be always enable
//    // Hence, there is no interfce to control error and alarm logs
//    void _debug(const char* text) {
//        string data;
//        data.append("[DEBUG]: ");
//        data.append(text);
//
//        if ((m_LogType == FILE_LOG) && (m_LogLevel >= DEBUG))
//        {
//            logIntoFile(data);
//        }
//        else if ((m_LogType == CONSOLE_LOG) && (m_LogLevel >= DEBUG))
//        {
//            logOnConsole(data);
//        }
//    }
//    
//
//    // Interfaces to control log levels
//    void updateLogLevel(LogLevel logLevel) {
//        m_LogLevel = logLevel;
//    }
//
//    // Enable all log levels
//    void enaleLog()
//    {
//        m_LogLevel = ALL_LOG;
//    }
//
//    // Disable all log levels, except error and alarm
//    void disableLog()
//    {
//        m_LogLevel = NO_LOG_LEVEL;
//    }
//
//    // Interfaces to control log Types
//    void updateLogType(LogType logType)
//    {
//        m_LogType = logType;
//    }
//
//    void enableConsoleLogging()
//    {
//        m_LogType = CONSOLE_LOG;
//    }
//
//    void enableFileLogging()
//    {
//        m_LogType = FILE_LOG;
//    }
//    
//    std::string getLogType() {
//        return _log_type;
//    }
//    std::string getLogLevel() {
//        return _log_level;
//    }
//    std::string getLogFname() {
//        return _log_fname;
//    }
//
//
//    MiniLoggerStream getLoggerStream(std::string message_type) {
//        MiniLoggerStream logger_stream(this);
//        return logger_stream;
//
//
//    }
//
//        
//    // Wrapper function for lock/unlock
//    // For Extensible feature, lock and unlock should be in protected
//    void lock(){
//        m_Mutex.lock();
//    }
//
//    void unlock() {
//        m_Mutex.unlock();
//    }
//
//    std::string getCurrentTime() {
//
//        string currTime;
//        //Current date/time based on current time
//        time_t now = time(0);
//        // Convert current time to string
//        currTime.assign(ctime(&now));
//
//        // Last charactor of currentTime is "\n", so remove it
//        string currentTime = currTime.substr(0, currTime.size() - 1);
//        return currentTime;
//    }
//
//    void logIntoFile(std::string& data) {
//        lock();
//        m_File << getCurrentTime() << "  " << data << endl;
//        m_File.flush();
//        unlock();
//
//    }
//    void logOnConsole(std::string& data) {
//        cout << getCurrentTime() << "  " << data << endl;
//    }
//    LogType stringToLogType(std::string log_type_str) {
//        std::map<std::string, LogType>::const_iterator mitr = log_type_map.find(log_type_str);
//        if (mitr != log_type_map.end()) {
//            return mitr->second;
//        }
//        else {
//            return CONSOLE_LOG;
//        }
//    }
//    LogLevel stringToLogLevel(std::string log_level_str) {
//        std::map<std::string, LogLevel>::const_iterator mitr = log_level_map.find(log_level_str);
//        if (mitr != log_level_map.end()) {
//            return mitr->second;
//        }
//        else {
//            return ALL_LOG;
//        }
//    }
//
//    MiniLoggerImpl(const MiniLoggerImpl& obj) {}
//    void operator=(const MiniLoggerImpl& obj) {}
//
//
//
//};
//
//
//
//
//
//
///// <summary>
///// //////////////////////////////////////////////////////////////////////////////////////////////////////
///// </summary>


//
//MiniLogger::MiniLogger():
//    p_logger(std::make_unique<MiniLoggerImpl>())
//{
//}
//
//void MiniLogger::initialize(std::string log_fname, std::string log_type, std::string log_level) {
//    
//    this->p_logger->initialize(log_fname, log_type, log_level);
//
//}
//
//MiniLogger::~MiniLogger()
//{
//    /*m_File.close();*/
//}
//
//
//template<>
//MiniLoggerStream CompuCell3D::operator<<(MiniLogger& logger, const LogMessageType &  val)
//{
//    MiniLoggerStream logger_stream(&logger);
//    logger_stream.setLogLevel(val);
//    return logger_stream;
//}
//
//MiniLogger* MiniLogger::getInstance() throw ()
//{
//    if (m_Instance == 0)
//    {
//        m_Instance = new MiniLogger();
//    }
//    return m_Instance;
//}
//
//std::string MiniLogger::getLogType() {
//    return this->p_logger->getLogType();    
//}
//
//std::string MiniLogger::getLogLevel() {
//    return this->p_logger->getLogLevel();    
//}
//
//std::string MiniLogger::getLogFname() {
//    return this->p_logger->getLogFname();
//}
//
//MiniLoggerStream MiniLogger::getLoggerStream(std::string message_type) {
//
//}
//
////void MiniLogger::lock()
////{
////    
////    m_Mutex.lock();
////}
////
////void MiniLogger::unlock()
////{
////    m_Mutex.unlock();
////}
//
//
////void MiniLogger::logOnConsole(std::string& data)
////{
////    cout << getCurrentTime() << "  " << data << endl;
////}
//
////LogType MiniLogger::stringToLogType(std::string log_type_str) {
////    //std::map<std::string, LogType>::const_iterator mitr = log_type_map.find(log_type_str);
////    //if (mitr != log_type_map.end()) {
////    //    return mitr->second;
////    //}
////    //else {
////    //    return CONSOLE_LOG;
////    //}
////}
//
////LogLevel MiniLogger::stringToLogLevel(std::string log_level_str) {
////    std::map<std::string, LogLevel>::const_iterator mitr = log_level_map.find(log_level_str);
////    if (mitr != log_level_map.end()) {
////        return mitr->second;
////    }
////    else {
////        return ALL_LOG;
////    }
////}
//
////string MiniLogger::getCurrentTime()
////{
////    string currTime;
////    //Current date/time based on current time
////    time_t now = time(0);
////    // Convert current time to string
////    currTime.assign(ctime(&now));
////
////    // Last charactor of currentTime is "\n", so remove it
////    string currentTime = currTime.substr(0, currTime.size() - 1);
////    return currentTime;
////}
//
////// Interface for Error Log
////void MiniLogger::_error(const char* text) throw()
////{
////    string data;
////    data.append("[ERROR]: ");
////    data.append(text);
////
////    // ERROR must be capture
////    if (m_LogType == FILE_LOG)
////    {
////        logIntoFile(data);
////    }
////    else if (m_LogType == CONSOLE_LOG)
////    {
////        logOnConsole(data);
////    }
////}
////
////// Interface for Alarm Log 
////void MiniLogger::_alarm(const char* text) throw()
////{
////    string data;
////    data.append("[ALARM]: ");
////    data.append(text);
////
////    // ALARM must be capture
////    if (m_LogType == FILE_LOG)
////    {
////        logIntoFile(data);
////    }
////    else if (m_LogType == CONSOLE_LOG)
////    {
////        logOnConsole(data);
////    }
////}
////
////// Interface for Always Log 
////void MiniLogger::_always(const char* text) throw()
////{
////    string data;
////    data.append("[ALWAYS]: ");
////    data.append(text);
////
////    // No check for ALWAYS logs
////    if (m_LogType == FILE_LOG)
////    {
////        logIntoFile(data);
////    }
////    else if (m_LogType == CONSOLE_LOG)
////    {
////        logOnConsole(data);
////    }
////}
////
////// Interface for Buffer Log 
////void MiniLogger::_buffer(const char* text) throw()
////{
////    // Buffer is the special case. So don't add log level
////    // and timestamp in the buffer message. Just log the raw bytes.
////    if ((m_LogType == FILE_LOG) && (m_LogLevel >= BUFFER))
////    {
////        lock();
////        m_File << text << endl;
////        unlock();
////    }
////    else if ((m_LogType == CONSOLE_LOG) && (m_LogLevel >= BUFFER))
////    {
////        cout << text << endl;
////    }
////}
////
////// Interface for Info Log
////void MiniLogger::_info(const char* text) throw()
////{
////    string data;
////    data.append("[INFO]: ");
////    data.append(text);
////
////    if ((m_LogType == FILE_LOG) && (m_LogLevel >= INFO))
////    {
////        logIntoFile(data);
////    }
////    else if ((m_LogType == CONSOLE_LOG) && (m_LogLevel >= INFO))
////    {
////        logOnConsole(data);
////    }
////}
////
////// Interface for Trace Log
////void MiniLogger::_trace(const char* text) throw()
////{
////    string data;
////    data.append("[TRACE]: ");
////    data.append(text);
////
////    if ((m_LogType == FILE_LOG) && (m_LogLevel >= TRACE))
////    {
////        logIntoFile(data);
////    }
////    else if ((m_LogType == CONSOLE_LOG) && (m_LogLevel >= TRACE))
////    {
////        logOnConsole(data);
////    }
////}
//
//// Interface for Debug Log
//void MiniLogger::_debug(const char* text) throw()
//{
//    this->p_logger->_debug(text);
//
//}
//
//// Interfaces to control log levels
//void MiniLogger::updateLogLevel(LogLevel logLevel)
//{
//    this->p_logger->updateLogLevel(logLevel);
//
//    
//}
//
//// Enable all log levels
//void MiniLogger::enaleLog()
//{
//    this->p_logger->enaleLog();
//    
//}
//
//// Disable all log levels, except error and alarm
//void MiniLogger::disableLog()
//{
//    this->p_logger->disableLog();
//    
//}
//
//// Interfaces to control log Types
//void MiniLogger::updateLogType(LogType logType)
//{
//    this->p_logger->updateLogType(logType);
//}
//
//void MiniLogger::enableConsoleLogging()
//{
//    this->p_logger->enableConsoleLogging();    
//}
//
//void MiniLogger::enableFileLogging()
//{
//    this->p_logger->enableFileLogging();
//}
//
