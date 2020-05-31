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
#include <iostream> 
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <unordered_map>
#include <mutex>
#include <functional> 

#include "Logger.h"

using namespace std;
using namespace CompuCell3D;



using namespace std;

Logger* Logger::m_Instance = 0;

Logger* Logger::getInstance() throw ()
{
    if (m_Instance == 0)
    {
        m_Instance = new Logger();
    }
    return m_Instance;
}

class Logger::Impl {
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

};




// Constructor connected with our Impl structure 
Logger::Logger()
    : pimpl(std::make_unique<Impl>())
{    }

void Logger::initialize(std::string log_fname, std::string log_type, std::string log_level) {
    this->pimpl->initialize(log_fname, log_type, log_level);
}

// Default Constructor 
Logger::~Logger() = default;

std::string Logger::getLogType() {
    return this->pimpl->getLogType();    
}

std::string Logger::getLogLevel() {
    return this->pimpl->getLogLevel();
}

std::string Logger::getLogFname() {
    return this->pimpl->getLogFname();
}

// Interfaces to control log levels
void Logger::updateLogLevel(LogLevel logLevel)
{
    this->pimpl->updateLogLevel(logLevel);   
}

// Enable all log levels
void Logger::enaleLog()
{
    this->pimpl->enaleLog();    
}

// Disable all log levels, except error and alarm
void Logger::disableLog()
{
    this->pimpl->disableLog();    
}

// Interfaces to control log Types
void Logger::updateLogType(LogType logType)
{
    this->pimpl->updateLogType(logType);
}

void Logger::enableConsoleLogging()
{
    this->pimpl->enableConsoleLogging();
}

void Logger::enableFileLogging()
{
    this->pimpl->enableFileLogging();
}

void  Logger::_debug(const char* text) {
    this->pimpl->_debug(text);
}

void Logger::_error(const char* text) throw(){
    this->pimpl->_error(text);
}

// Interface for Alarm Log 
void Logger::_alarm(const char* text) throw() {
    this->pimpl->_alarm(text);
}
// Interface for Always Log 
void Logger::_always(const char* text) throw() {
    this->pimpl->_always(text);
}
// Interface for Buffer Log 
void Logger::_buffer(const char* text) throw() {
    this->pimpl->_buffer(text);
}
// Interface for Info Log 
void Logger::_info(const char* text) throw() {
    this->pimpl->_info(text);
}

// Interface for Trace log 
void Logger::_trace(const char* text) throw() {
    this->pimpl->_trace(text);
}


template<>
LoggerStream CompuCell3D::operator<<(Logger& logger, const LogMessageType &  val) {
    LoggerStream logger_stream(&logger);
    logger_stream.setLogLevel(val);
    return logger_stream;
}


class LoggerStream::StreamImpl {
public:


    using logger_fcn_t = std::function<void(Logger *, std::string&)>;

    StreamImpl(Logger *logger_p) {

        this->logger_p = logger_p;

    }

    ~StreamImpl() {
        using namespace std;
        try {
            logger_fcn_map.at(this->logMessageType)(this->logger_p, this->logString);
        }
        catch (const out_of_range &e)
        {
            cerr << "Exception in at method while logging " << e.what() << endl;
        }


    }

    void setLogLevel(const LogMessageType & logMessageType = LogMessageType::DEBUG_LOG) {
        this->logMessageType = logMessageType;
    }

    void addString(std::string & str) {
        this->logString += str;
    }
    typedef double (LoggerStream::*log_function_t)(std::string & text);

private:

    std::unordered_map<LogMessageType, logger_fcn_t> logger_fcn_map = {
        { LogMessageType::DEBUG_LOG , [](Logger * logger_p, std::string& text) { logger_p->_debug(text.c_str()); } },
        { LogMessageType::ERROR_LOG , [](Logger * logger_p, std::string& text) { logger_p->_error(text.c_str()); } },
        { LogMessageType::INFO_LOG , [](Logger * logger_p, std::string& text) { logger_p->_info(text.c_str()); } },
        { LogMessageType::TRACE_LOG , [](Logger * logger_p, std::string& text) { logger_p->_trace(text.c_str()); } },
        { LogMessageType::ALARM_LOG , [](Logger * logger_p, std::string& text) { logger_p->_alarm(text.c_str()); } },
        { LogMessageType::BUFFER_LOG , [](Logger * logger_p, std::string& text) { logger_p->_buffer(text.c_str()); } },

    };

private:
    Logger *logger_p;
    std::string logString;
    LogMessageType  logMessageType;

};


LoggerStream::LoggerStream(Logger *logger_p): 
    pimpl(std::make_unique<StreamImpl>(logger_p))
{}

LoggerStream::LoggerStream(const LoggerStream & rhs) :
    pimpl(std::make_unique<StreamImpl>(*rhs.pimpl))
{

}

LoggerStream::~LoggerStream() = default;

void LoggerStream::addString(std::string & str) {
    this->pimpl->addString(str);
}

void LoggerStream::setLogLevel(const LogMessageType & logMessageType) {
    this->pimpl->setLogLevel(logMessageType);
}



