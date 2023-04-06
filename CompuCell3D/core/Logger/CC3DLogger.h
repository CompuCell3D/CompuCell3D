/**
 * @file CC3DLogger.h
 * @author Saumya Mehta.
 * @brief Defines the logger; Adapted from Mechanica CC3DLogger.h 
 * originally written by T.J Sego PhD; apparently taken from libRoadRunner rrLogger
 * @date 2021-07-04
 * 
 */
#pragma once
#ifndef SRC_CC3DLOGGER_H_
#define SRC_CC3DLOGGER_H_

#ifdef CC3D_ISWIN
#   define CC3D_FUNCTION __func__
#else
#   define CC3D_FUNCTION __PRETTY_FUNCTION__
#endif

#include <sstream>
#include <fstream>
#include "LoggerDLLSpecifier.h"


/**
 * Poco LogStream dumps to the log when a newline i.e. std::endl is encountered,
 * however the old proprietary logging system dumps basically when the stream object
 * goes out of scope.
 *
 * This object allows us to to use the new Poco logging system and maintain
 * compatability with all existing code.
 *
 * This object is returned from the rr::Logger, exposes a ostream interface, and
 * and dumps to the log when it goes out of scope.
 */
namespace CompuCell3D {

    class Message {
    public:
        enum Priority
        {
            PRIO_FATAL       = 1,
            PRIO_CRITICAL,
            PRIO_ERROR,
            PRIO_WARNING,
            PRIO_NOTICE,
            PRIO_INFORMATION,
            PRIO_DEBUG,
            PRIO_TRACE,
        };
    };




    class LOGGER_EXPORT LoggingBuffer {
    public:
        LoggingBuffer(int level, const char *func, const char *file, int line);

        /**
         * dump the contents of the stringstream to the log.
         */
        ~LoggingBuffer();

        /**
         * get the stream this buffer holds.
         */
        std::ostream &stream();

    private:
        std::stringstream buffer;
        int level;
        const char *func;
        const char *file;
        int line;
    };

/**
 * same as Poco level, repeat here to avoid including any Poco files
 * as Poco is usually linked statically so third parties would not need
 * to have Poco installed.
 */
    enum LogLevel {
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

    enum LogEvent {
        LOG_OUTPUTSTREAM_CHANGED,
        LOG_LEVEL_CHANGED,
        LOG_CALLBACK_SET
    };

    typedef int (*LoggerCallback)(LogEvent, std::ostream *);


    class LOGGER_EXPORT CC3DLogger {
    private:

        static CC3DLogger *singleton;
        int logLevel = Message::PRIO_NOTICE;
        std::ofstream outputFile;
        std::string outputFileName;
        std::ostream *consoleStream = nullptr;
        std::ostream *fileStream = nullptr;
        CompuCell3D::LoggerCallback callback = nullptr;
		CC3DLogger(){};


    public:

        static CC3DLogger *get();
        static void destroy();
        
        CC3DLogger(const CC3DLogger &) = delete;
		void operator=(const CC3DLogger &) = delete;


        /**
         * @brief Set the Level object sets the logging level to one a value from Logger::Level
         *
         * @param level logging level
         */
        void setLevel(int level = LOG_CURRENT);

        /**
         * @brief Get the Level objectget the current logging level.
         *
         * @return int
         */
        int getLevel();

        /**
         * @brief Suppresses all logging output
         */
        void disableLogging();

        /**
         * @brief stops logging to the console, but file logging may continue.
         */
        void disableConsoleLogging();

        /**
         * @brief turns on console logging at the given level.
         *
         * @param level logging level
         */
        void enableConsoleLogging(int level = LOG_CURRENT);

        /**
         * @brief turns on file logging to the given file as the given level.
         *
         * If fileName is an empty string, then nothing occurs.
         *
         * @param fileName path to log file
         * @param level logging level
         */
        void enableFileLogging(const std::string &fileName = "",
                               int level = LOG_CURRENT);

        /**
         * @brief turns off file logging, but has no effect on console logging.
         *
         */
        void disableFileLogging();

        /**
         * @brief get the textural form of the current logging level.
         *
         * @return std::string
         */
        std::string getCurrentLevelAsString();

        /**
         * @brief Get the File Name objectget the name of the currently used log file.
         *
         * @return std::string
         */
        std::string getFileName();

        /**
         * gets the textual form of a logging level Enum for a given value.
         */
        static std::string levelToString(int level);

        /**
         * parses a string and returns a Logger::Level
         */
        static LogLevel stringToLevel(const std::string &str);

        /**
         * @brief logs a message to the log.
         *
         * @param level logging level
         * @param msg log message
         */
        void log(LogLevel level, const std::string &msg);


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
        void setConsoleStream(std::ostream *os);
        std::ostream * getConsoleStream();

        std::ostream * getFileStream();



        void setCallback(LoggerCallback);

    };

}

#define CC3D_Log(level) \
    if (level > CompuCell3D::CC3DLogger::get()->getLevel()) { ; } \
    else CompuCell3D::LoggingBuffer(level, CC3D_FUNCTION, __FILE__, __LINE__).stream()

#endif /* SRC_CC3DLOGGER_H_ */