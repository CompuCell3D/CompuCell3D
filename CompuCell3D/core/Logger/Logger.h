///////////////////////////////////////////////////////////////////////////////
// @File Name:     Logger.h                                                  //
// @Author:        Pankaj Choudhary                                          //
// @Version:       0.0.1                                                     //
// @L.M.D:         13th April 2015                                           //
// @Description:   For Logging into file          
// @Adapted by: Maciek Swat
// @M.M.D          29th May 2020 
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

#pragma once 

#include <memory> // PImpl 
#include <string> 
#include <sstream>


namespace  CompuCell3D
{

    template<typename T, typename... Args>
    std::unique_ptr<T> make_unique(Args&&... args)
    {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }

    enum class LogLevel
    {
        NO_LOG_LEVEL = 1,
        INFO = 2,
        BUFFER = 3,
        TRACE = 4,
        DEBUG = 5,
        ALL_LOG = 6,
    };
    
        
    enum class LogType
    {
        NO_LOG = 1,
        CONSOLE_LOG = 2,
        FILE_LOG = 3,
    };
    
    
    enum class LogMessageType
    {
        DEBUG_LOG,
        ERROR_LOG,        
        INFO_LOG,
        TRACE_LOG,
        ALARM_LOG,        
        BUFFER_LOG
    };


class Logger {
public:
    // Constructor and Destructors 
    static Logger* getInstance() throw ();

    // Asssignment Operator and Copy Constructor 

    void initialize(std::string log_fname, std::string log_type = "file_log", std::string log_level = "enable_log");

    void _error(const char* text) throw();

    // Interface for Alarm Log 
    void _alarm(const char* text) throw();

    // Interface for Always Log 
    void _always(const char* text) throw();

    // Interface for Buffer Log 
    void _buffer(const char* text) throw();

    // Interface for Info Log 
    void _info(const char* text) throw();

    // Interface for Trace log 
    void _trace(const char* text) throw();

    // Interface for Debug log 
    void _debug(const char* text) throw();


    // Interfaces to control log levels
    void updateLogLevel(LogLevel logLevel);
    void enaleLog();  // Enable all log levels
    void disableLog(); // Disable all log levels, except error and alarm
    
    // Interfaces to control log Types
    void updateLogType(LogType logType);
    void enableConsoleLogging();
    void enableFileLogging();

    std::string getLogType();
    std::string getLogLevel();
    std::string getLogFname();



private:
    // Internal implementation class 
    class Impl;
    // Pointer to the internal implementation 
    std::unique_ptr<Impl> pimpl;
    ~Logger();
    Logger();

    Logger(const Logger& other);
    Logger& operator=(Logger rhs);

    static Logger * m_Instance;

};


class LoggerStream {
public:

        
    LoggerStream(Logger *logger_p);
    LoggerStream(const LoggerStream & rhs);

    ~LoggerStream();

    void setLogLevel(const LogMessageType & logMessageType = LogMessageType::DEBUG_LOG);

    void addString(std::string & str);
    

private:

    class StreamImpl;
    // Pointer to the internal implementation 
    std::unique_ptr<StreamImpl> pimpl;

};


template<typename T>
LoggerStream operator<<(Logger & logger, const T & val) {
    LoggerStream logger_stream(&logger);

    logger_stream << val;

    return logger_stream;
}

//specialization for stream modifier LogLevel -  implementation must be defined in implementation file
template<>
LoggerStream operator<<(Logger & logger, const LogMessageType &  val);



// T && is necessary when we are chaining operators because then we might be passing rvalue references (temporaries)
// use this or const T & - some compilers are picky how you define this. MSVS for example is not picky and T & works
template<typename T>
LoggerStream & operator<<(LoggerStream  & loggerStream, T && val) {
    using namespace std;
    std::ostringstream s_stream;
    s_stream << val << " ";
    std::string str = s_stream.str();
    loggerStream.addString(str);
    return loggerStream;

};

// in c++11 it looks like for certain compilers we need to provide specialized version of that takes universal reference
// to handle situations where e.g. log<<a<<"demo"; we do chaining and as a consequence compiler is looking for
// a signature where we could pass LoggerStream temporary object (rvalue)
// https://stackoverflow.com/questions/41216362/c-passing-rvalue-reference-to-functions-that-takes-lvalue-reference
template<typename T>
LoggerStream & operator<<(LoggerStream  && loggerStream, T && val) {
    using namespace std;
    std::ostringstream s_stream;
    s_stream << val << " ";
    std::string str = s_stream.str();
    loggerStream.addString(str);
    return loggerStream;

};


}


