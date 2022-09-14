/**
 * @file Logger.cpp
 * @author Saumya Mehta.
 * @brief Defines the logger; derived from mechanica Logger.h originally written by T.J Sego, PhD; apparently taken from libRoadRunner rrLogger
 * @date 2022-07-08
 * 
 */
#include "CC3DLogger.h"
#include <iostream>
#include <algorithm>
#include <fstream>

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

static int logLevel = Message::PRIO_NOTICE;

static std::ofstream outputFile;
static std::string outputFileName;

static std::ostream *consoleStream = NULL;
static std::ostream *fileStream = NULL;

static CompuCell3D::LoggerCallback callback = NULL;



class FakeLogger {
public:
    void fatal(const std::string& fmt, const char* func, const char* file, const int line);
    void critical(const std::string& fmt, const char* func, const char* file, const int line);
    void error(const std::string& fmt, const char* func, const char* file, const int line);
    void warning(const std::string& fmt, const char* func, const char* file, const int line);
    void notice(const std::string& fmt, const char* func, const char* file, const int line);
    void information(const std::string& fmt, const char* func, const char* file, const int line);
    void debug(const std::string& fmt, const char* func, const char* file, const int line);
    void trace(const std::string& fmt, const char* func, const char* file, const int line);
};

static FakeLogger& getLogger() {
    static FakeLogger logger;
    return logger;
}

CompuCell3D::LoggingBuffer::LoggingBuffer(int level, const char* func, const char *file, int line):
                func(func), file(file), line(line)
{
    if (level >= Message::PRIO_FATAL && level <= Message::PRIO_TRACE)
    {
        this->level = level;
    }
    else
    {
        // wrong level, so just set to error?
        this->level = Message::PRIO_ERROR;
    }
}

CompuCell3D::LoggingBuffer::~LoggingBuffer()
{
    FakeLogger &logger = getLogger();
    switch (level)
    {
    case Message::PRIO_FATAL:
        logger.fatal(buffer.str(), func, file, line);
        break;
    case Message::PRIO_CRITICAL:
        logger.critical(buffer.str(), func, file, line);
        break;
    case Message::PRIO_ERROR:
        logger.error(buffer.str(), func, file, line);
        break;
    case Message::PRIO_WARNING:
        logger.warning(buffer.str(), func, file, line);
        break;
    case Message::PRIO_NOTICE:
        logger.notice(buffer.str(), func, file, line);
        break;
    case Message::PRIO_INFORMATION:
        logger.information(buffer.str(), func, file, line);
        break;
    case Message::PRIO_DEBUG:
        logger.debug(buffer.str(), func, file, line);
        break;
    case Message::PRIO_TRACE:
        logger.trace(buffer.str(), func, file, line);
        break;
    default:
        logger.error(buffer.str(), func, file, line);
        break;
    }
}

std::ostream& CompuCell3D::LoggingBuffer::stream()
{
    return buffer;
}

void CompuCell3D::Logger::setLevel(int level)
{
    logLevel = level;
    
    if(callback) {
        if (consoleStream) callback(LOG_LEVEL_CHANGED, consoleStream);
        if (fileStream) callback(LOG_LEVEL_CHANGED, fileStream);
    }
}

int CompuCell3D::Logger::getLevel()
{
    return logLevel;
}

void CompuCell3D::Logger::disableLogging()
{
    disableConsoleLogging();
    disableFileLogging();
}

void CompuCell3D::Logger::disableConsoleLogging()
{
    consoleStream = NULL;
    if(callback) callback(LOG_OUTPUTSTREAM_CHANGED, consoleStream);
}

void CompuCell3D::Logger::enableConsoleLogging(int level)
{
    setLevel(level);

    consoleStream = &std::cout;

    if(callback) {
        callback(LOG_OUTPUTSTREAM_CHANGED, consoleStream);
    }
}

void CompuCell3D::Logger::enableFileLogging(const std::string &fileName, int level)
{
    setLevel(level);

    disableFileLogging();

    outputFileName = fileName;
    outputFile.open(fileName, std::ios_base::out|std::ios_base::ate);
    if(outputFile.is_open()) {
        fileStream = &outputFile;
    }

    if(callback) {
        callback(LOG_OUTPUTSTREAM_CHANGED, fileStream);
    }
}

void CompuCell3D::Logger::disableFileLogging()
{
    if (outputFileName.size() == 0) return;

    outputFile.close();
    outputFileName = "";
    fileStream = NULL;

    if(callback) {
        callback(LOG_OUTPUTSTREAM_CHANGED, fileStream);
    }
}

std::string CompuCell3D::Logger::getCurrentLevelAsString()
{
    return levelToString(logLevel);
}

std::string CompuCell3D::Logger::getFileName()
{
    return outputFileName;
}

// void Logger::setFormattingPattern(const std::string &format)
// {
// }

// std::string Logger::getFormattingPattern()
// {
//     return "";
// }

std::string CompuCell3D::Logger::levelToString(int level)
{
    switch (level)
    {
    case Message::PRIO_FATAL:
        return "LOG_FATAL";
        break;
    case Message::PRIO_CRITICAL:
        return "LOG_CRITICAL";
        break;
    case Message::PRIO_ERROR:
        return "LOG_ERROR";
        break;
    case Message::PRIO_WARNING:
        return "LOG_WARNING";
        break;
    case Message::PRIO_NOTICE:
        return "LOG_NOTICE";
        break;
    case Message::PRIO_INFORMATION:
        return "LOG_INFORMATION";
        break;
    case Message::PRIO_DEBUG:
        return "LOG_DEBUG";
        break;
    case Message::PRIO_TRACE:
        return "LOG_TRACE";
        break;
    default:
        return "LOG_CURRENT";
    }
    return "LOG_CURRENT";
}

CompuCell3D::LogLevel CompuCell3D::Logger::stringToLevel(const std::string &str)
{
    std::string upstr = str;
    std::transform(upstr.begin(), upstr.end(), upstr.begin(), ::toupper);

    if (upstr == "LOG_FATAL")
    {
        return LOG_FATAL;
    }
    else if(upstr == "LOG_CRITICAL")
    {
        return LOG_CRITICAL;
    }
    else if(upstr == "LOG_ERROR" || upstr == "ERROR")
    {
        return LOG_ERROR;
    }
    else if(upstr == "LOG_WARNING" || upstr == "WARNING")
    {
        return LOG_WARNING;
    }
    else if(upstr == "LOG_NOTICE")
    {
        return LOG_NOTICE;
    }
    else if(upstr == "LOG_INFORMATION" || upstr == "INFO")
    {
        return LOG_INFORMATION;
    }
    else if(upstr == "LOG_DEBUG" || upstr == "DEBUG")
    {
        return LOG_DEBUG;
    }
    else if(upstr == "LOG_TRACE" || upstr == "TRACE")
    {
        return LOG_TRACE;
    }
    else
    {
        return LOG_CURRENT;
    }
}

// bool Logger::getColoredOutput()
// {
//     return false;
// }

// void Logger::setColoredOutput(bool bool1)
// {
// }

// void Logger::setProperty(const std::string &name, const std::string &value)
// {
// }

void CompuCell3D::Logger::log(LogLevel l, const std::string &msg)
{
    FakeLogger &logger = getLogger();

    Message::Priority level = (Message::Priority)(l);

    switch (level)
    {
    case Message::PRIO_FATAL:
            logger.fatal(msg, "", "", 0);
        break;
    case Message::PRIO_CRITICAL:
            logger.critical(msg, "", "", 0);
        break;
    case Message::PRIO_ERROR:
            logger.error(msg, "", "", 0);
        break;
    case Message::PRIO_WARNING:
            logger.warning(msg, "", "", 0);
        break;
    case Message::PRIO_NOTICE:
            logger.notice(msg, "", "", 0);
        break;
    case Message::PRIO_INFORMATION:
            logger.information(msg, "", "", 0);
        break;
    case Message::PRIO_DEBUG:
            logger.debug(msg, "", "", 0);
        break;
    case Message::PRIO_TRACE:
            logger.trace(msg, "", "", 0);
        break;
    default:
            logger.error(msg, "", "", 0);
        break;
    }
}

void CompuCell3D::Logger::setConsoleStream(std::ostream *os)
{
    consoleStream = os;

    if(callback) callback(LOG_OUTPUTSTREAM_CHANGED, consoleStream);
}

void CompuCell3D::Logger::setCallback(LoggerCallback cb) {
    callback = cb;
    
    if(callback) {
        if(consoleStream) callback(LOG_CALLBACK_SET, consoleStream);
        if(fileStream) callback(LOG_CALLBACK_SET, fileStream);
    }
}

static void write_log(const char* kind, const std::string &fmt, const char* func, const char *file, const int line, std::ostream *os) {
    
    *os << kind << ": " << fmt;
    if(func) { *os << ", func: " << func;}
    if(file) {*os << ", file:" << file;}
    if(line >= 0) {*os << ",lineno:" << line;}
    *os << std::endl;
}

static void write_log(const char* kind, const std::string &fmt, const char* func, const char *file, const int line) {
    
    if(consoleStream) write_log(kind, fmt, func, file, line, consoleStream);
    if(fileStream) write_log(kind, fmt, func, file, line, fileStream);
}

void FakeLogger::fatal(const std::string &fmt, const char* func, const char *file,
        const int line)
{
    write_log("FATAL", fmt, func, file, line);
}

void FakeLogger::critical(const std::string &fmt, const char* func, const char *file,
        const int line)
{
    write_log("CRITICAL", fmt, func, file, line);
}

void FakeLogger::error(const std::string &fmt, const char* func, const char *file,
        const int line)
{
    write_log("ERROR", fmt, func, file, line);
}

void FakeLogger::warning(const std::string &fmt, const char* func, const char *file,
        const int line)
{
    write_log("WARNING", fmt, func, file, line);
}

void FakeLogger::notice(const std::string &fmt, const char* func, const char *file,
        const int line)
{
    write_log("NOTICE", fmt, func, file, line);
}

void FakeLogger::information(const std::string &fmt, const char* func, const char *file,
        const int line)
{
    write_log("INFO", fmt, func, file, line);
}

void FakeLogger::debug(const std::string &fmt, const char* func, const char *file,
        const int line)
{
    write_log("DEBUG", fmt, func, file, line);
}

void FakeLogger::trace(const std::string &fmt, const char* func, const char *file,
        const int line)
{
    write_log("TRACE", fmt, func, file, line);
}