CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
project(poco_foundation_src)

# Option that allows users to build static and/or shared libraries
# ----------------------------------------------------------------
OPTION(POCO_BUILD_STATIC_LIBS "Build static libraries" ON)
OPTION(POCO_BUILD_SHARED_LIBS "Build shared libraries" OFF)
set(target poco_foundation) 

set(poco_include_folder ${CMAKE_CURRENT_SOURCE_FOLDER}../include)

set(sources 
ASCIIEncoding
Ascii
AtomicCounter
AbstractObserver
ActiveDispatcher
ArchiveStrategy
AsyncChannel
Base64Decoder
Base64Encoder
BinaryReader
BinaryWriter
Bugcheck
ByteOrder
Channel
Checksum
Condition
Configurable
ConsoleChannel
CountingStream
DateTime
DateTimeFormat
DateTimeFormatter
DateTimeParser
Debugger
DeflatingStream
DigestEngine
DigestStream
DirectoryIterator
Environment
ErrorHandler
Event
EventArgs
Exception
FPEnvironment
File
#FIFOBufferStream
FileChannel
FileStream
FileStreamFactory
Format
Formatter
FormattingChannel
Glob
Hash
HashStatistic
HexBinaryDecoder
HexBinaryEncoder
InflatingStream
Latin1Encoding
#Latin2Encoding
Latin9Encoding
LineEndingConverter
LocalDateTime
LogFile
LogStream
Logger
LoggingFactory
LoggingRegistry
MD4Engine
MD5Engine
Manifest
MemoryPool
MemoryStream
Message
Mutex
NamedEvent
NamedMutex
NestedDiagnosticContext
Notification
NotificationCenter
NotificationQueue
TimedNotificationQueue
PriorityNotificationQueue
NullChannel
NullStream
NumberFormatter
NumberParser
Path
PatternFormatter
Pipe
PipeImpl
PipeStream
Process
PurgeStrategy
RWLock
Random
RandomStream
RefCountedObject
RegularExpression
RotateStrategy
Runnable
SHA1Engine
Semaphore
SharedLibrary
SharedMemory
SignalHandler
SimpleFileChannel
SplitterChannel
Stopwatch
StreamChannel
StreamConverter
StreamCopier
StreamTokenizer
String
StringTokenizer
SynchronizedObject
Task
TaskManager
TaskNotification
TeeStream
TemporaryFile
TextConverter
TextEncoding
TextIterator
TextBufferIterator
Thread
ThreadTarget
ThreadLocal
ThreadPool
Timer
Timespan
Timestamp
Timezone
Token
URI
URIStreamFactory
URIStreamOpener
Unicode
UnicodeConverter
UTF8Encoding
UTF16Encoding
Windows1252Encoding
TextEncoding
Environment
UUID
UUIDGenerator
Void
deflate.c
zutil.c
adler32.c
crc32.c
trees.c
)

if(${BORLAND})
add_definitions(
             -w-8012                     #Comparing signed /unsigned
             -w-8057                     #Parameter never used
             -w-8004                     #'var' is assigned a value that is never used
             -w-8123                     #Path not found
             -w-8066                     #Unreachable code
             -w-8104                     #Local static with constructor...
             -w-8008                     #Condition is always false..
        )
add_definitions( -D_WIN32)
endif()

#if(CMAKE_SYSTEM MATCHES "Windows")
# 
# add_definitions( -DPOCO_OS_FAMILY_WINDOWS)
# set(SYSLIBS iphlpapi gdi32 odbc32)
# 
# if (CMAKE_C_COMPILER_ID MATCHES "MSVC")
#        message(STATUS "XXX: MS Visual Compiler detected")
# endif (CMAKE_C_COMPILER_ID MATCHES "MSVC")
#
#endif(CMAKE_SYSTEM MATCHES "Windows")
#
#if (CMAKE_SYSTEM MATCHES "Linux")
#  add_definitions( -DPOCO_OS_FAMILY_UNIX )
#  # Standard 'must be' defines
#  add_definitions( -D_XOPEN_SOURCE=500 -D_REENTRANT -D_THREAD_SAFE -D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64)
#  set(SYSLIBS  pthread dl rt)
#endif(CMAKE_SYSTEM MATCHES "Linux")

if(${MSVC})
#set(....
endif()

include_directories(
${poco_include_folder}
)
#add_library(${target} SHARED ${sources})
#
#set_property(TARGET ${target}  
#             PROPERTY  COMPILE_DEFINITIONS 
#                       WIN32
#                       SHARED_NLEQ
#             ) 
#
#Then the static version
add_library(${target}-static STATIC ${sources})
set_property(TARGET ${target}-static  
             PROPERTY  COMPILE_DEFINITIONS 
                       POCO_STATIC
                       PCRE_STATIC
             ) 

if(WIN32)
set_property(TARGET ${target}-static  
             PROPERTY  COMPILE_DEFINITIONS 
                       WIN32
             ) 

endif()

#INSTALLING ======================
FILE (GLOB headers ${poco_include_folder}/Poco/*.h)

#Install required files
install(FILES ${headers} 
DESTINATION include/Poco
)

#install(TARGETS ${target} 
#        LIBRARY DESTINATION lib 
#        ARCHIVE DESTINATION lib
#        RUNTIME DESTINATION bin)
#
install(TARGETS ${target}-static 
        LIBRARY DESTINATION lib 
        ARCHIVE DESTINATION lib)

