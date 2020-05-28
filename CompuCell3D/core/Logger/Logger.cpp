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

// Code Specific Header Files(s)
#include "Logger.h"

using namespace std;
using namespace CompuCell3D;

Logger* Logger::m_Instance = 0;


Logger::Logger()
{
}

void Logger::initialize(std::string log_fname, std::string log_type, std::string log_level) {
    
    _log_level = log_level;
    _log_type = log_type;
    _log_fname = log_fname;
    m_LogType = stringToLogType(log_type);
    m_LogLevel = stringToLogLevel(log_level);


    if (log_fname.size()) {
        
        if(m_LogType == FILE_LOG) {
            m_File.open(log_fname, ios::out | ios::app);
        }
        else {
            m_LogType = CONSOLE_LOG;
        }
    }

}

Logger::~Logger()
{
    m_File.close();
}


template<>
LoggerStream CompuCell3D::operator<<(Logger& logger, const LogLevel & val)
{
    LoggerStream logger_stream(&logger);
    logger_stream.setLogLevel(val);
    //logger_stream << val;

    return logger_stream;
}


//LoggerStream CompuCell3D::operator<<(Logger& logger, int const & number) {
//    cerr << "got number " << number << endl;
//    LoggerStream logger_stream(&logger);
//    return logger_stream;
//}

//LoggerStream&  CompuCell3D::operator<<(LoggerStream& logger, int const & number) {
//    cerr << "LOGGER STREAM got number " << number << endl;
//    return logger;
//}

//void CompuCell3D::printDemo(int number) {
//    cerr << "got number " << number << endl;
//}

//int Logger::operator<< (int & number) {
//    cerr << "got number " << number << endl;
//}

Logger* Logger::getInstance() throw ()
{
    if (m_Instance == 0)
    {
        m_Instance = new Logger();
    }
    return m_Instance;
}

std::string Logger::getLogType() {
    return _log_type;
}

std::string Logger::getLogLevel() {
    return _log_level;
}

std::string Logger::getLogFname() {
    return _log_fname;
}

LoggerStream Logger::getLoggerStream(std::string message_type) {
    LoggerStream logger_stream(this);
    return logger_stream;
}

void Logger::lock()
{
    m_Mutex.lock();
}

void Logger::unlock()
{
    m_Mutex.unlock();
}

void Logger::logIntoFile(std::string& data)
{
    lock();
    m_File << getCurrentTime() << "  " << data << endl;
    unlock();
}

void Logger::logOnConsole(std::string& data)
{
    cout << getCurrentTime() << "  " << data << endl;
}

LogType Logger::stringToLogType(std::string log_type_str) {
    std::map<std::string, LogType>::const_iterator mitr = log_type_map.find(log_type_str);
    if (mitr != log_type_map.end()) {
        return mitr->second;
    }
    else {
        return CONSOLE_LOG;
    }
}

LogLevel Logger::stringToLogLevel(std::string log_level_str) {
    std::map<std::string, LogLevel>::const_iterator mitr = log_level_map.find(log_level_str);
    if (mitr != log_level_map.end()) {
        return mitr->second;
    }
    else {
        return ALL_LOG;
    }
}

string Logger::getCurrentTime()
{
    string currTime;
    //Current date/time based on current time
    time_t now = time(0);
    // Convert current time to string
    currTime.assign(ctime(&now));

    // Last charactor of currentTime is "\n", so remove it
    string currentTime = currTime.substr(0, currTime.size() - 1);
    return currentTime;
}

// Interface for Error Log
void Logger::_error(const char* text) throw()
{
    string data;
    data.append("[ERROR]: ");
    data.append(text);

    // ERROR must be capture
    if (m_LogType == FILE_LOG)
    {
        logIntoFile(data);
    }
    else if (m_LogType == CONSOLE_LOG)
    {
        logOnConsole(data);
    }
}

void Logger::_error(std::string& text) throw()
{
    _error(text.data());
}

void Logger::_error(std::ostringstream& stream) throw()
{
    string text = stream.str();
    _error(text.data());
}

// Interface for Alarm Log 
void Logger::_alarm(const char* text) throw()
{
    string data;
    data.append("[ALARM]: ");
    data.append(text);

    // ALARM must be capture
    if (m_LogType == FILE_LOG)
    {
        logIntoFile(data);
    }
    else if (m_LogType == CONSOLE_LOG)
    {
        logOnConsole(data);
    }
}

void Logger::_alarm(std::string& text) throw()
{
    _alarm(text.data());
}

void Logger::_alarm(std::ostringstream& stream) throw()
{
    string text = stream.str();
    _alarm(text.data());
}

// Interface for Always Log 
void Logger::_always(const char* text) throw()
{
    string data;
    data.append("[ALWAYS]: ");
    data.append(text);

    // No check for ALWAYS logs
    if (m_LogType == FILE_LOG)
    {
        logIntoFile(data);
    }
    else if (m_LogType == CONSOLE_LOG)
    {
        logOnConsole(data);
    }
}

void Logger::_always(std::string& text) throw()
{
    _always(text.data());
}

void Logger::_always(std::ostringstream& stream) throw()
{
    string text = stream.str();
    _always(text.data());
}

// Interface for Buffer Log 
void Logger::_buffer(const char* text) throw()
{
    // Buffer is the special case. So don't add log level
    // and timestamp in the buffer message. Just log the raw bytes.
    if ((m_LogType == FILE_LOG) && (m_LogLevel >= BUFFER))
    {
        lock();
        m_File << text << endl;
        unlock();
    }
    else if ((m_LogType == CONSOLE_LOG) && (m_LogLevel >= BUFFER))
    {
        cout << text << endl;
    }
}

void Logger::_buffer(std::string& text) throw()
{
    _buffer(text.data());
}

void Logger::_buffer(std::ostringstream& stream) throw()
{
    string text = stream.str();
    _buffer(text.data());
}

// Interface for Info Log
void Logger::_info(const char* text) throw()
{
    string data;
    data.append("[INFO]: ");
    data.append(text);

    if ((m_LogType == FILE_LOG) && (m_LogLevel >= INFO))
    {
        logIntoFile(data);
    }
    else if ((m_LogType == CONSOLE_LOG) && (m_LogLevel >= INFO))
    {
        logOnConsole(data);
    }
}

void Logger::_info(std::string& text) throw()
{
    _info(text.data());
}

void Logger::_info(std::ostringstream& stream) throw()
{
    string text = stream.str();
    _info(text.data());
}

// Interface for Trace Log
void Logger::_trace(const char* text) throw()
{
    string data;
    data.append("[TRACE]: ");
    data.append(text);

    if ((m_LogType == FILE_LOG) && (m_LogLevel >= TRACE))
    {
        logIntoFile(data);
    }
    else if ((m_LogType == CONSOLE_LOG) && (m_LogLevel >= TRACE))
    {
        logOnConsole(data);
    }
}

void Logger::_trace(std::string& text) throw()
{
    _trace(text.data());
}

void Logger::_trace(std::ostringstream& stream) throw()
{
    string text = stream.str();
    _trace(text.data());
}

// Interface for Debug Log
void Logger::_debug(const char* text) throw()
{
    string data;
    data.append("[DEBUG]: ");
    data.append(text);

    if ((m_LogType == FILE_LOG) && (m_LogLevel >= DEBUG))
    {
        logIntoFile(data);
    }
    else if ((m_LogType == CONSOLE_LOG) && (m_LogLevel >= DEBUG))
    {
        logOnConsole(data);
    }
}

void Logger::_debug(std::string& text) throw()
{
    _debug(text.data());
}

void Logger::_debug(std::ostringstream& stream) throw()
{
    string text = stream.str();
    _debug(text.data());
}

// Interfaces to control log levels
void Logger::updateLogLevel(LogLevel logLevel)
{
    m_LogLevel = logLevel;
}

// Enable all log levels
void Logger::enaleLog()
{
    m_LogLevel = ALL_LOG;
}

// Disable all log levels, except error and alarm
void Logger::disableLog()
{
    m_LogLevel = NO_LOG_LEVEL;
}

// Interfaces to control log Types
void Logger::updateLogType(LogType logType)
{
    m_LogType = logType;
}

void Logger::enableConsoleLogging()
{
    m_LogType = CONSOLE_LOG;
}

void Logger::enableFileLogging()
{
    m_LogType = FILE_LOG;
}

