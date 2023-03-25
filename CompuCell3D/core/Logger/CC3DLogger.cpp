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

using namespace std;
using namespace CompuCell3D;

CC3DLogger *CC3DLogger::singleton;


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

LoggingBuffer::LoggingBuffer(int level, const char* func, const char *file, int line):
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

LoggingBuffer::~LoggingBuffer()
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

std::ostream& LoggingBuffer::stream()
{
    return buffer;
}


CC3DLogger* CC3DLogger::get() {
    using namespace std;
    if (!singleton) {

        singleton = new CC3DLogger();
    }

    return singleton;
}

void CC3DLogger::destroy() {

    if (singleton) {

        delete singleton;
        singleton = 0;
    }
}


void CC3DLogger::setLevel(int level)
{
    logLevel = level;

    if(callback) {
        if (consoleStream) callback(LOG_LEVEL_CHANGED, consoleStream);
        if (fileStream) callback(LOG_LEVEL_CHANGED, fileStream);
    }
}

int CC3DLogger::getLevel()
{
    return logLevel;
}

void CC3DLogger::disableLogging()
{
    disableConsoleLogging();
    disableFileLogging();
}

void CC3DLogger::disableConsoleLogging()
{
    consoleStream = NULL;
    if(callback) callback(LOG_OUTPUTSTREAM_CHANGED, consoleStream);
}

void CC3DLogger::enableConsoleLogging(int level)
{
    setLevel(level);

    consoleStream = &std::cout;

    if(callback) {
        callback(LOG_OUTPUTSTREAM_CHANGED, consoleStream);
    }
}

void CC3DLogger::enableFileLogging(const std::string &fileName, int level)
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


void CC3DLogger::disableFileLogging()
{
    if (outputFileName.size() == 0) return;

    outputFile.close();
    outputFileName = "";
    fileStream = nullptr;

    if(callback) {
        callback(LOG_OUTPUTSTREAM_CHANGED, fileStream);
    }
}

std::string CC3DLogger::getCurrentLevelAsString()
{
    return levelToString(logLevel);
}

std::string CC3DLogger::getFileName()
{
    return outputFileName;
}



std::string CC3DLogger::levelToString(int level)
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

LogLevel CC3DLogger::stringToLevel(const std::string &str)
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

void CC3DLogger::log(LogLevel l, const std::string &msg)
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

void CC3DLogger::setConsoleStream(std::ostream *os)
{
    consoleStream = os;

    if(callback) callback(LOG_OUTPUTSTREAM_CHANGED, consoleStream);
}

std::ostream * CC3DLogger::getConsoleStream(){
    return consoleStream;
}


std::ostream * CC3DLogger::getFileStream(){
    return fileStream;
}

void CC3DLogger::setCallback(LoggerCallback cb) {
    callback = cb;

    if(callback) {
        if(consoleStream) callback(LOG_CALLBACK_SET, consoleStream);
        if(fileStream) callback(LOG_CALLBACK_SET, fileStream);
    }
}

//void Logger::setLevel(int level)
//{
//    logLevel = level;
//
//    if(callback) {
//        if (consoleStream) callback(LOG_LEVEL_CHANGED, consoleStream);
//        if (fileStream) callback(LOG_LEVEL_CHANGED, fileStream);
//    }
//}

//int Logger::getLevel()
//{
//    return logLevel;
//}

//void Logger::disableLogging()
//{
//    disableConsoleLogging();
//    disableFileLogging();
//}

//void Logger::disableConsoleLogging()
//{
//    consoleStream = NULL;
//    if(callback) callback(LOG_OUTPUTSTREAM_CHANGED, consoleStream);
//}

//void Logger::enableConsoleLogging(int level)
//{
//    setLevel(level);
//
//    consoleStream = &std::cout;
//
//    if(callback) {
//        callback(LOG_OUTPUTSTREAM_CHANGED, consoleStream);
//    }
//}

//void Logger::enableFileLogging(const std::string &fileName, int level)
//{
//    setLevel(level);
//
//    disableFileLogging();
//
//    outputFileName = fileName;
//    outputFile.open(fileName, std::ios_base::out|std::ios_base::ate);
//    if(outputFile.is_open()) {
//        fileStream = &outputFile;
//    }
//
//    if(callback) {
//        callback(LOG_OUTPUTSTREAM_CHANGED, fileStream);
//    }
//}

//void Logger::disableFileLogging()
//{
//    if (outputFileName.size() == 0) return;
//
//    outputFile.close();
//    outputFileName = "";
//    fileStream = NULL;
//
//    if(callback) {
//        callback(LOG_OUTPUTSTREAM_CHANGED, fileStream);
//    }
//}

//std::string Logger::getCurrentLevelAsString()
//{
//    return levelToString(logLevel);
//}

//std::string Logger::getFileName()
//{
//    return outputFileName;
//}

// void Logger::setFormattingPattern(const std::string &format)
// {
// }

// std::string Logger::getFormattingPattern()
// {
//     return "";
// }
//
//std::string Logger::levelToString(int level)
//{
//    switch (level)
//    {
//    case Message::PRIO_FATAL:
//        return "LOG_FATAL";
//        break;
//    case Message::PRIO_CRITICAL:
//        return "LOG_CRITICAL";
//        break;
//    case Message::PRIO_ERROR:
//        return "LOG_ERROR";
//        break;
//    case Message::PRIO_WARNING:
//        return "LOG_WARNING";
//        break;
//    case Message::PRIO_NOTICE:
//        return "LOG_NOTICE";
//        break;
//    case Message::PRIO_INFORMATION:
//        return "LOG_INFORMATION";
//        break;
//    case Message::PRIO_DEBUG:
//        return "LOG_DEBUG";
//        break;
//    case Message::PRIO_TRACE:
//        return "LOG_TRACE";
//        break;
//    default:
//        return "LOG_CURRENT";
//    }
//    return "LOG_CURRENT";
//}
//
//LogLevel Logger::stringToLevel(const std::string &str)
//{
//    std::string upstr = str;
//    std::transform(upstr.begin(), upstr.end(), upstr.begin(), ::toupper);
//
//    if (upstr == "LOG_FATAL")
//    {
//        return LOG_FATAL;
//    }
//    else if(upstr == "LOG_CRITICAL")
//    {
//        return LOG_CRITICAL;
//    }
//    else if(upstr == "LOG_ERROR" || upstr == "ERROR")
//    {
//        return LOG_ERROR;
//    }
//    else if(upstr == "LOG_WARNING" || upstr == "WARNING")
//    {
//        return LOG_WARNING;
//    }
//    else if(upstr == "LOG_NOTICE")
//    {
//        return LOG_NOTICE;
//    }
//    else if(upstr == "LOG_INFORMATION" || upstr == "INFO")
//    {
//        return LOG_INFORMATION;
//    }
//    else if(upstr == "LOG_DEBUG" || upstr == "DEBUG")
//    {
//        return LOG_DEBUG;
//    }
//    else if(upstr == "LOG_TRACE" || upstr == "TRACE")
//    {
//        return LOG_TRACE;
//    }
//    else
//    {
//        return LOG_CURRENT;
//    }
//}

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

//void Logger::log(LogLevel l, const std::string &msg)
//{
//    FakeLogger &logger = getLogger();
//
//    Message::Priority level = (Message::Priority)(l);
//
//    switch (level)
//    {
//    case Message::PRIO_FATAL:
//            logger.fatal(msg, "", "", 0);
//        break;
//    case Message::PRIO_CRITICAL:
//            logger.critical(msg, "", "", 0);
//        break;
//    case Message::PRIO_ERROR:
//            logger.error(msg, "", "", 0);
//        break;
//    case Message::PRIO_WARNING:
//            logger.warning(msg, "", "", 0);
//        break;
//    case Message::PRIO_NOTICE:
//            logger.notice(msg, "", "", 0);
//        break;
//    case Message::PRIO_INFORMATION:
//            logger.information(msg, "", "", 0);
//        break;
//    case Message::PRIO_DEBUG:
//            logger.debug(msg, "", "", 0);
//        break;
//    case Message::PRIO_TRACE:
//            logger.trace(msg, "", "", 0);
//        break;
//    default:
//            logger.error(msg, "", "", 0);
//        break;
//    }
//}
//
//void Logger::setConsoleStream(std::ostream *os)
//{
//    consoleStream = os;
//
//    if(callback) callback(LOG_OUTPUTSTREAM_CHANGED, consoleStream);
//}
//
//void Logger::setCallback(LoggerCallback cb) {
//    callback = cb;
//
//    if(callback) {
//        if(consoleStream) callback(LOG_CALLBACK_SET, consoleStream);
//        if(fileStream) callback(LOG_CALLBACK_SET, fileStream);
//    }
//}

void write_log(const char* kind, const std::string &fmt, const char* func, const char *file, const int line, std::ostream *os) {
    
    *os << kind << ": " << fmt;
    if(func) { *os << ", func: " << func;}
    if(file) {*os << ", file:" << file;}
    if(line >= 0) {*os << ",lineno:" << line;}
    *os << std::endl;
}

void write_log(const char* kind, const std::string &fmt, const char* func, const char *file, const int line) {
    std::ostream *consoleStream = CC3DLogger::get()->getConsoleStream();
    std::ostream *fileStream =  CC3DLogger::get()->getFileStream();
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
//    write_log("INFO", fmt, func, file, line);
    write_log("INFO", fmt, 0, 0, -1);
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