/**
 * @file CC3DLogger.h
 * @author Saumya Mehta.
 * @brief Defines the logger; derived from mechanica CC3DLogger.h originally written by T.J Sego PhD; apparently taken from libRoadRunner rrLogger
 * @date 2021-07-04
 * 
 */
#pragma once
#ifndef SRC_CC3DLOGGER_H_
#define SRC_CC3DLOGGER_H_

#include <sstream>

#ifdef CC3D_POCO
namespace Poco {
class Logger;
}
#endif


/**
 * Poco LogStream dumps to the log when a newline i.e. std::endl is encountered,
 * howeve the old proprietary logging system dumps basically when the stream object
 * goes out of scope.
 *
 * This object allows us to to use the new Poco logging system and maintain
 * compatability with all existing code.
 *
 * This object is returned from the rr::Logger, exposes a ostream interface, and
 * and dumps to the log when it goes out of scope.
 */
class CAPI_EXPORT CC3DLoggingBuffer
{
public:
    CC3DLoggingBuffer(int level, const char* func, const char* file, int line);

    /**
     * dump the contents of the stringstream to the log.
     */
    ~CC3DLoggingBuffer();

    /**
     * get the stream this buffer holds.
     */
    std::ostream &stream();

private:
    std::stringstream buffer;
    int level;
    const char* func;
    const char* file;
    int line;
};

/**
 * same as Poco level, repeat here to avoid including any Poco files
 * as Poco is usually linked statically so third parties would not need
 * to have Poco installed.
 */
enum CC3DLogLevel
{
    LOG_CURRENT = 0, ///< Use the current level -- don't change the level from what it is.
    LOG_FATAL = 1,   ///< A fatal error. The application will most likely terminate. This is the highest priority.
    LOG_CRITICAL,    ///< A critical error. The application might not be able to continue running successfully.
    LOG_ERROR,       ///< An error. An operation did not complete successfully, but the application as a whole is not affected.
    LOG_WARNING,     ///< A warning. An operation completed with an unexpected result.
    LOG_NOTICE,      ///< A notice, which is an information with just a higher priority.
    LOG_INFORMATION, ///< An informational message, usually denoting the successful completion of an operation.
    LOG_DEBUG,       ///< A debugging message.
    LOG_TRACE        ///< A tracing message. This is the lowest priority.
};

enum CC3DLogEvent
{
    LOG_OUTPUTSTREAM_CHANGED,
    LOG_LEVEL_CHANGED,
    LOG_CALLBACK_SET
};

typedef HRESULT (*CC3DLoggerCallback)(CC3DLogEvent, std::ostream *);

/**
 * The Compucell3D logger.
 *
 * A set of static method for setting the logging level.
 */
class CAPI_EXPORT CC3DLogger
{
public:

    /**
     * @brief Set the Level objectsets the logging level to one a value from Logger::Level
     * 
     * @param level logging level
     */
    static void setLevel(int level = LOG_CURRENT);

    /**
     * @brief Get the Level objectget the current logging level.
     * 
     * @return int 
     */
    static int getLevel();

    /**
     * @brief Suppresses all logging output
     */
    static void disableLogging();

    /**
     * @brief stops logging to the console, but file logging may continue.
     */
    static void disableConsoleLogging();

    /**
     * @brief turns on console logging at the given level.
     * 
     * @param level logging level
     */
    static void enableConsoleLogging(int level = LOG_CURRENT);

    /**
     * @brief turns on file logging to the given file as the given level.
     * 
     * If fileName is an empty string, then nothing occurs. 
     * 
     * @param fileName path to log file
     * @param level logging level
     */
    static void enableFileLogging(const std::string& fileName = "",
            int level = LOG_CURRENT);

    /**
     * @brief turns off file logging, but has no effect on console logging.
     * 
     */
    static void disableFileLogging();

    /**
     * @brief get the textural form of the current logging level.
     * 
     * @return std::string 
     */
    static std::string getCurrentLevelAsString();

    /**
     * @brief Get the File Name objectget the name of the currently used log file.
     * 
     * @return std::string 
     */
    static std::string getFileName();

    /**
     * gets the textual form of a logging level Enum for a given value.
     */
    static std::string levelToString(int level);

    /**
     * parses a string and returns a Logger::Level
     */
    static CC3DLogLevel stringToLevel(const std::string& str);

    /**
     * @brief logs a message to the log.
     * 
     * @param level logging level
     * @param msg log message
     */
    static void log(CC3DLogLevel level, const std::string& msg);


    /**
     * Set a pointer to an ostream object where the console logger should
     * log to.
     *
     * Normally, this points to std::clog.
     *
     * This is here so that the Logger can properly re-direct to the
     * Python sys.stderr object as the QT IPython console only
     * reads output from the python sys.stdout and sys.stderr
     * file objects and not the C++ file streams.
     */
    static void setConsoleStream(std::ostream *os);


    static void setCallback(CC3DLoggerCallback);

};

#define Log(level) \
    if (level > CC3DLogger::getLevel()) { ; } \
    else CC3DLoggingBuffer(level, CC3D_FUNCTION, __FILE__, __LINE__).stream()

#endif /* SRC_CC3DLOGGER_H_ */