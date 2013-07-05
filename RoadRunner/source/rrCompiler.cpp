#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include <sstream>
#if defined(WIN32)
#include <windows.h>
#include <strsafe.h>
#if defined(__CODEGEARC__)
    #include <dir.h>
#elif defined(_MSVC)
    #include <direct.h>
#endif
#endif
#include "Poco/File.h"
#include "rrLogger.h"
#include "rrCompiler.h"
#include "rrException.h"
#include "rrStringUtils.h"
#include "rrUtils.h"
#include "rrRoadRunner.h"
//---------------------------------------------------------------------------

using namespace std;
namespace rr
{

Compiler::Compiler(const string& supportCodeFolder, const string& compiler)
:
mSupportCodeFolder(supportCodeFolder),
mCompilerName(ExtractFileName(compiler)),
mCompilerLocation(ExtractFilePath(compiler))
{
	if(mSupportCodeFolder.size() > 0)
    {
        if(!setupCompiler(mSupportCodeFolder))
        {
            Log(lWarning)<<"Roadrunner internal compiler setup failed. ";
        }
    }
}

Compiler::~Compiler(){}

bool Compiler::setupCompiler(const string& supportCodeFolder)
{
    mSupportCodeFolder = supportCodeFolder;

    if(!FolderExists(mSupportCodeFolder))
    {
    	Log(lError)<<"The roadrunner support code folder : "<<mSupportCodeFolder<<" does not exist.";
        return false;
    }

    return true;
}

bool Compiler::setOutputPath(const string& path)
{
	mOutputPath = path;
	return true;
}

bool Compiler::compileSource(const string& sourceFileName)
{
    //Compile the code and load the resulting dll, and call an exported function in it...
#if defined(_WIN32) || defined(__CODEGEARC__)
    string dllFName(ChangeFileExtensionTo(ExtractFileName(sourceFileName), "dll"));
// match __unix__ and __APPLE__ separately (from Andy S.)  --- :
#elif defined(__unix__)
    string dllFName(ChangeFileExtensionTo(ExtractFileName(sourceFileName), "so"));
#elif defined(__APPLE__)
    string dllFName(ChangeFileExtensionTo(ExtractFileName(sourceFileName), "dylib"));
#endif
    mDLLFileName = JoinPath(ExtractFilePath(sourceFileName), dllFName);

	cerr<<"mDLLFileName ="<<mDLLFileName <<endl;
	////avoiding compilation of a file if it exists
	//if (FileExists(mDLLFileName)){
	//	return true;
	//}

    //Setup compiler environment
    setupCompilerEnvironment();

    string exeCmd = createCompilerCommand(sourceFileName);
	cerr<<"exeCmd ="<<exeCmd <<endl;

    //exeCmd += " > compileLog.log";
    Log(lDebug2)<<"Compiling model..";
    Log(lDebug)<<"\nExecuting compile command: "<<exeCmd;

    if(!compile(exeCmd))
    {
        Log(lError)<<"Creating DLL failed..";
        throw Exception("Creating Model DLL failed..");
    }

    //Check if the DLL exists...
    return FileExists(mDLLFileName);
}

bool Compiler::setCompiler(const string& compiler)
{
	mCompilerName = ExtractFileName(compiler);
	mCompilerLocation = ExtractFilePath(compiler);
	return true;
}

bool Compiler::setCompilerLocation(const string& path)
{
	if(!FolderExists(path))
	{
		Log(lError)<<"Tried to set invalid path: "<<path<<" for compiler location";
		return false;
	}
	mCompilerLocation = path;
	return true;
}

string	Compiler::getCompilerLocation()
{
	return mCompilerLocation;
}

bool Compiler::setSupportCodeFolder(const string& path)
{
	if(!FolderExists(path))
	{
		Log(lError)<<"Tried to set invalid path: "<<path<<" for compiler location";
		return false;
	}
	mSupportCodeFolder = path;
	return true;
}

string	Compiler::getSupportCodeFolder()
{
	return mSupportCodeFolder;
}

bool Compiler::setupCompilerEnvironment()
{
    mIncludePaths.clear();
    mLibraryPaths.clear();
    mCompilerFlags.clear();
    if(ExtractFileNameNoExtension(mCompilerName) == "tcc" || ExtractFileNameNoExtension(mCompilerName) == "gcc")
    {
        mCompilerFlags.push_back("-g");         //-g adds runtime debug information
// match __unix__ and _WIN32 separately from __APPLE__ (from Andy S.)  --- :
#if defined(__unix__) || defined(_WIN32)
        mCompilerFlags.push_back("-shared");
        mCompilerFlags.push_back("-rdynamic");  //-rdynamic : Export global symbols to the dynamic linker
#elif defined(__APPLE__)
        mCompilerFlags.push_back("-dynamiclib");
#endif
                                                //-b : Generate additional support code to check memory allocations and array/pointer bounds. `-g' is implied.

        mCompilerFlags.push_back("-fPIC"); // shared lib
        mCompilerFlags.push_back("-O0"); // turn off optimization

        //LogLevel                              //-v is for verbose
        if(ExtractFileNameNoExtension(mCompilerName) == "tcc")
        {
            mIncludePaths.push_back(".");
            mIncludePaths.push_back("r:/rrl/source");

            mIncludePaths.push_back(JoinPath(mCompilerLocation, "include"));
            mLibraryPaths.push_back(".");
            mLibraryPaths.push_back(JoinPath(mCompilerLocation, "lib"));
            if(gLog.GetLogLevel() < lDebug)
            {
                mCompilerFlags.push_back("-v"); // suppress warnings
            }
            else if(gLog.GetLogLevel() >= lDebug1)
            {
                mCompilerFlags.push_back("-vv");
            }
            else if(gLog.GetLogLevel() >= lDebug2)
            {
                mCompilerFlags.push_back("-vvv");
            }
        }
        else if(ExtractFileNameNoExtension(mCompilerName) == "gcc")
        {
            if(gLog.GetLogLevel() < lDebug)
            {
                mCompilerFlags.push_back("-w"); // suppress warnings
            }
            else if(gLog.GetLogLevel() >= lDebug1)
            {
                mCompilerFlags.push_back("-Wall");
            }
            else if(gLog.GetLogLevel() >= lDebug2)
            {
                mCompilerFlags.push_back("-Wall -pedantic");
            }
        }
    }

    mIncludePaths.push_back(mSupportCodeFolder);
    return true;
}

string Compiler::createCompilerCommand(const string& sourceFileName)
{
    stringstream exeCmd;
    // also check for "cc" (from Andy S.)  --- :
//    if(ExtractFileNameNoExtension(mCompilerName) == "tcc" || ExtractFileNameNoExtension(mCompilerName) == "gcc")
    if(ExtractFileNameNoExtension(mCompilerName) == "tcc" 
       || ExtractFileNameNoExtension(mCompilerName) == "gcc"
       || ExtractFileNameNoExtension(mCompilerName) == "cc")
    {
        exeCmd<<JoinPath(mCompilerLocation, mCompilerName);
        //Add compiler flags
        for(int i = 0; i < mCompilerFlags.size(); i++)
        {
            exeCmd<<" "<<mCompilerFlags[i];
        }
        exeCmd<<" \""<<sourceFileName<<"\" \""<<JoinPath(mSupportCodeFolder, "rrSupport.c")<<"\"";


        exeCmd<<" -o\""<<mDLLFileName<<"\"";
#if defined(WIN32)
		exeCmd<<" -DBUILD_MODEL_DLL ";
#endif
        //Add include paths
        for(int i = 0; i < mIncludePaths.size(); i++)
        {
            exeCmd<<" -I\""<<mIncludePaths[i]<<"\" " ;
        }

        //Add library paths
        for(int i = 0; i < mLibraryPaths.size(); i++)
        {
            exeCmd<<" -L\""<<mLibraryPaths[i]<<"\" " ;
        }
    }
    return exeCmd.str();
}

#ifdef WIN32

bool Compiler::compile(const string& cmdLine)
{
    if( !cmdLine.size() )
    {
        return false;
    }

    PROCESS_INFORMATION pi;
    ZeroMemory( &pi, sizeof(pi) );

    STARTUPINFO si;
    ZeroMemory( &si, sizeof(si) );
    si.cb = sizeof(si);

    //sec attributes for the output file
    SECURITY_ATTRIBUTES sao;
    sao.nLength=sizeof(SECURITY_ATTRIBUTES);
    sao.lpSecurityDescriptor=NULL;
    sao.bInheritHandle=1;

    string compilerTempFile(JoinPath(mOutputPath, ExtractFileNameNoExtension(mDLLFileName)));
    compilerTempFile.append("C.log");

    Poco::File aFile(compilerTempFile);
    if(aFile.exists())
    {
    	aFile.remove();
    }

    HANDLE outFile;
  	//Todo: there is a problem creating the logfile after first time creation..
    if((outFile=CreateFileA(compilerTempFile.c_str(),
                            GENERIC_WRITE,
                            FILE_SHARE_DELETE,
                            &sao,
                            OPEN_ALWAYS,
                            FILE_ATTRIBUTE_NORMAL,
                            NULL))==INVALID_HANDLE_VALUE)
    {
        // Retrieve the system error message for the last-error code
        DWORD errorCode = GetLastError();
        string anError = GetWINAPIError(errorCode, TEXT("CreateFile"));
        Log(lError)<<"WIN API Error (after CreateFile): "<<anError;
        Log(lError)<<"Failed creating logFile for compiler output";
    }

    SetFilePointer(outFile, 0, NULL, FILE_END); //set pointer position to end file

    //init the STARTUPINFO struct
    si.dwFlags=STARTF_USESTDHANDLES;
    si.hStdOutput = outFile;
    si.hStdError  = outFile;

    //proc sec attributes
    SECURITY_ATTRIBUTES sap;
    sap.nLength=sizeof(SECURITY_ATTRIBUTES);
    sap.lpSecurityDescriptor=NULL;
    sap.bInheritHandle=1;

    //thread sec attributes
    SECURITY_ATTRIBUTES sat;
    sat.nLength=sizeof(SECURITY_ATTRIBUTES);
    sat.lpSecurityDescriptor=NULL;
    sat.bInheritHandle=1;

    // Start the child process.
    if( !CreateProcessA(
        NULL,                           // No module name (use command line)
        (char*) cmdLine.c_str(),        // Command line
        &sap,                           // Process handle not inheritable
        &sat,                           // Thread handle not inheritable
        TRUE,                          // Set handle inheritance
        CREATE_NO_WINDOW,               // Creation flags
        NULL,                           // Use parent's environment block
        NULL,                           // Use parent's starting directory
        &si,                            // Pointer to STARTUPINFO structure
        &pi )                           // Pointer to PROCESS_INFORMATION structure
    )
    {
		DWORD errorCode = GetLastError();

        string anError = GetWINAPIError(errorCode, TEXT("CreateProcess"));
        Log(lError)<<"WIN API Error: (after CreateProcess) "<<anError;

        // Close process and thread handles.
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
        CloseHandle(outFile);
        return false;
    }

    // Wait until child process exits.
    WaitForSingleObject(pi.hProcess, INFINITE);

    CloseHandle(outFile);

    // Close process and thread handles.
    CloseHandle(pi.hProcess);
    DWORD errorCode = GetLastError();
    if(errorCode != 0)
    {
    	string anError = GetWINAPIError(errorCode, TEXT("CloseHandle"));
    	Log(lDebug)<<"WIN API error: (pi.hProcess)"<<anError;
    }

    CloseHandle(pi.hThread);
    errorCode = GetLastError();
    if(errorCode != 0)
    {
    	string anError = GetWINAPIError(errorCode, TEXT("CloseHandle"));
    	Log(lDebug)<<"WIN API error: (pi.hThread)"<<anError;
    }

    //Read the log file and log it
    if(FileExists(compilerTempFile))
    {
    	string log = GetFileContent(compilerTempFile.c_str());
    	Log(lDebug)<<"Compiler output: "<<log<<endl;
    }

    return true;
}

#else  //---------------- LINUX, UNIXES

bool Compiler::compile(const string& cmdLine)
{
    string toFile(cmdLine);
    toFile += " >> ";
    toFile += JoinPath(mOutputPath, "compilation.log");
    toFile += " 2>&1";

    Log(lDebug)<<"Compiler command: "<<toFile;

    //Create the shared library, using system call
    int val = system(toFile.c_str());
    if(val == 0)
    {
    	Log(lDebug)<<"Compile system call was succesful";
        return true;
    }
    else
    {
	    Log(lError)<<"Compile system call returned: "<<val;
        return false;
    }
}

#endif //WIN32
string getCompilerMessages()
{
    return "No messages yet";
}

} //namespace rr

