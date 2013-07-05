#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop

#if defined(WIN32)
#include <windows.h>
#include <io.h>
#include <conio.h>
#include <time.h>

#endif

#if defined(__BORLANDC__)
#include <dir.h>
#elif defined(_MSC_VER)
#include <direct.h>
#else
#include <unistd.h>
#include <sys/stat.h>
#include <limits.h>  //PATH_MAX
#endif

#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <math.h>
#include <float.h>
#if defined(WIN32)
#include <strsafe.h> //StringCchPrintf need to be included AFTER all other system headers :(
#endif
#include "Poco/MD5Engine.h"
#include "Poco/Thread.h"
#include "rrStringUtils.h"
#include "rrUtils.h"
#include "rrLogger.h"
#include "rrMisc.h"

// (from Andy S.)  --- :
// Most Unix systems have a getch in libcurses, but this introduces
// an un-needed depencency, as we can write our own getch easily.
// We do need the standard Posix termio headers though.
#if defined (__unix__) || defined(__APPLE__)
#include <stdlib.h>
#include <termios.h>
#endif

#if defined(__APPLE__)
#include <limits.h>  //PATH_MAX
#include <mach-o/dyld.h>
#endif


//---------------------------------------------------------------------------
namespace rr
{
using namespace std;
using namespace Poco;


// (from Andy S.)  --- :
// A function to get a character from the console without echo.
// equivalent of Windows / Curses getch function. Note, that the 
// curses library has the same thing, but not all systems have curses, 
// and makes no sense have a dependency on it for one simple function. 
#if defined(__unix__) || defined(__APPLE__)
static char rrGetch()
{
    char ch;
    termios _old, _new;

    /* Initialize new terminal i/o settings */
    tcgetattr(0, &_old); /* grab old terminal i/o settings */
    _new = _old; /* make new settings same as old settings */
    _new.c_lflag &= ~ICANON; /* disable buffered i/o */
    _new.c_lflag &= ~ECHO; /* set no echo mode */
    tcsetattr(0, TCSANOW, &_new); /* use these new terminal i/o settings now */

    ch = getchar();

    /* Restore old terminal i/o settings */
    tcsetattr(0, TCSANOW, &_old);
    return ch;
}
#elif defined (_WIN32)
// Windows has get built into conio
#define rrGetch getch
#endif

string getMD5(const string& text)
{
    MD5Engine md5;
    md5.update(text);
    string digestString(Poco::DigestEngine::digestToHex(md5.digest()));
    return digestString;
}

string getCurrentDateTime()
{
// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // Visit http://www.cplusplus.com/reference/clibrary/ctime/strftime/
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d %X", &tstruct);

    return buf;
}

string GetUsersTempDataFolder()
{
    //Default for temporary data output is the users AppData/Local/Temp Folder
    //  Gets the temp path env string (no guarantee it's a valid path).
#if defined(WIN32)
    TCHAR lpTempPathBuffer[MAX_PATH];
    DWORD dwRetVal = GetTempPathA(MAX_PATH, lpTempPathBuffer); // buffer for path
    if (dwRetVal > MAX_PATH || (dwRetVal == 0))
    {
        Log(lError)<<"GetTempPath failed";
    }
    else
    {
        Log(lDebug3)<<"Users temporary files folder is: "<<string(lpTempPathBuffer);
    }

    return string(lpTempPathBuffer);
#else
return ".";
#endif
}


string getCurrentExeFolder()
{
#if defined(_WIN32) || defined(WIN32)
	char path[MAX_PATH];
	if(GetModuleFileNameA(NULL, path, ARRAYSIZE(path)) != 0)
    {
	    string aPath(ExtractFilePath(path));
		return aPath;
    }
    return "";
// (from Andy S.)  --- :
#elif defined(__APPLE__)
    char path[PATH_MAX+1];
    unsigned  bufsize = sizeof(path);
    if (_NSGetExecutablePath(path, &bufsize) == 0) {
	    string aPath(ExtractFilePath(path));
		return aPath;
    }
#elif defined __linux
        char arg1[20];
        char exepath[PATH_MAX + 1] = {0};

        sprintf( arg1, "/proc/%d/exe", getpid() );
        readlink( arg1, exepath, 1024 );
		string thePath = ExtractFilePath(exepath); 
		Log(lDebug1)<<"Current exe folder says:"<<thePath;
        return thePath;
#endif

}

string getParentFolder(const string& path)
{
	if(path.size() < 1)
    {
    	return "";
    }
	vector<string> fldrs = SplitString(path, gPathSeparator);
    string parent("");
    if(fldrs.size() > 1)
	{
    	for(int i = 0; i < fldrs.size() -1; i++)
        {
			parent = JoinPath(parent, fldrs[i]);
        }

        string pathSep;
        pathSep.push_back(gPathSeparator);
        if(path.compare(0,1, pathSep) == 0)
		{
			parent = gPathSeparator + parent;
		}
		return parent;
    }
    else
    {
    	return path;
    }
}

string getCWD()
{
    //Get the working directory
	char *buffer;

	string cwd;
	// Get the current working directory:
	if( (buffer = getcwd( NULL, 0 )) == NULL )
	{
		Log(lError)<<"getCWD failed";
		return "";
	}
	else
	{
      cwd = buffer;
      free(buffer);
	}

	return cwd;
}

void sleep(int ms)
{
    Poco::Thread::sleep(10);
}

const char getPathSeparator()
{
	return gPathSeparator;
}

string GetFileContent(const string& fName)
{
	string content;

    vector<string> lines = GetLinesInFile(fName);
    for(int i = 0; i < lines.size(); i++)
    {
        content += lines[i];
        content += "\n";
    }

    return content;
}

vector<string> GetLinesInFile(const string& fName)
{
    vector<string> lines;

    ifstream ifs(fName.c_str());
    if(!ifs)
    {
        Log(lError)<<"Failed opening file: "<<fName;
        return lines;
    }

    std::string oneLine((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    lines = SplitString(oneLine, "\n");
    return lines;
}

std::size_t IndexOf(std::vector<std::string>& vec, const std::string& elem )
{
    int index = distance(vec.begin(), find(vec.begin(), vec.end(), elem));
    return index;
}

// String utils
//string RemoveTrailingSeparator(const string& fldr, const char sep = gPathSeparator);//"\\");
string RemoveTrailingSeparator(const string& _folder, const char sep)
{
    if((_folder.size() > 0) && (_folder[_folder.size() -1] == sep))
    {
        const int endOfPathIndex = _folder.rfind(sep, _folder.size());
        string folder = _folder.substr(0, endOfPathIndex);
        return folder;
    }
    else
    {
        return _folder;
    }
}

bool IsNaN(const double& aNum)
{
#if defined(WIN32)
    return _isnan(aNum) > 0 ? true : false;
#else
    return std::isnan(aNum) > 0 ? true : false;
#endif
}

bool IsNullOrEmpty(const string& str)
{
    return !str.size();
}

void Pause(bool doIt, const string& msg)
{
    if(!doIt)
    {
        return;
    }

    if(msg.size() == 0)
    {
    	cout<<"Hit any key to exit...";
    }
    else
    {
    	cout<<msg;
    }
    cin.ignore(0,'\n');

    // (from Andy S.)  --- :
    // On Windows this just calls the built in getch.
    rrGetch();
}

bool FileExists(const string& fName)
{
    if (!fName.size())
    {
        return false;
    }
    
#if defined(__linux)
    ifstream test(fName);
    return test;
#else

    bool res = (access(fName.c_str(), 0) == 0);
    return res;
#endif    
}

bool FolderExists(const string& folderName)
{
#if defined(WIN32)
    LPCTSTR szPath = folderName.c_str();
    DWORD dwAttrib = GetFileAttributes(szPath);
    return (dwAttrib != INVALID_FILE_ATTRIBUTES && (dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
#else
    struct stat status;
    return stat(folderName.c_str(), &status) == 0 ? true : false;
#endif
}

void CreateTestSuiteFileNameParts(int caseNr, const string& postFixPart, string& modelFilePath, string& modelName, string& settingsFName)
{
    stringstream modelSubPath;
    stringstream modelFileName;
    stringstream settingsFileName;

    modelSubPath<<setfill('0')<<setw(5)<<caseNr;        //create the "00023" subfolder format
    modelFileName<<setfill('0')<<setw(5)<<caseNr<<postFixPart;
    modelFilePath = JoinPath(modelFilePath, modelSubPath.str());
    modelName =  modelFileName.str();
    settingsFileName <<setfill('0')<<setw(5)<<caseNr<<"-settings.txt";
	settingsFName = settingsFileName.str();
}

string GetTestSuiteSubFolderName(int caseNr)
{
    stringstream modelSubPath;
    modelSubPath<<setfill('0')<<setw(5)<<caseNr;        //create the "00023" subfolder format
    return modelSubPath.str();
}

bool CreateFolder(const string& folder)
{
    if(FileExists(folder))
    {
        return true;
    }

#if defined(WIN32)
    int res = mkdir(folder.c_str());
#else
	int temp;
#define MY_MASK 0777
// 	printf("Default mask: %o\n", MY_MASK & ~022 & MY_MASK);
  	temp = umask(0);
//  	printf("Previous umask = %o\n", temp);
    int res = mkdir(folder.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
#endif

    return (res==0) ? true : false;
}

bool CreateFile(const string& fName, std::ios_base::openmode mode)
{
    ofstream test;
    test.open(fName.c_str(), mode);
    test.close();
    return FileExists(fName);
}

bool CopyValues(vector<double>& dest, double* source, const int& nrVals, const int& startIndex)
{
    if(!dest.size() || !source || startIndex > dest.size())
    {
        return false;
    }

    for(int i = startIndex; i < nrVals; i++)
    {
        dest[i] = source[i];
    }

    return true;

}

bool CopyStdVectorToCArray(const vector<double>& src, double* dest,  int size)
{
    if(!dest || size > src.size())
    {
        Log(lError)<<"Tried to copy to NULL vector, or incompatible size of vectors";
        return false;
    }

    for(int i = 0; i < size; i++)
    {
        dest[i] = src[i];
    }
    return true;
}

bool CopyStdVectorToCArray(const vector<bool>&   src,  bool*  dest,  int size)
{
    if(!dest || size > src.size())
    {
        Log(lError)<<"Tried to copy to NULL vector, or incompatible size of vectors";
        return false;
    }


    for(int i = 0; i < size; i++)
    {
        dest[i] = src[i];
    }
    return true;
}

vector<double> CreateVector(const double* src, const int& size)
{
    vector<double> dest;
    if(!src)
    {
        Log(lError)<<"Tried to copy from NULL vector";
        return dest;
    }

    dest.resize(size);
    for(int i = 0; i < size; i++)
    {
        dest[i] = src[i];
    }
    return dest;
}

bool CopyCArrayToStdVector(const int* src, vector<int>& dest, int size)
{
    if(!src)
    {
        Log(lError)<<"Tried to copy from NULL vector";
        return false;
    }

    dest.resize(size);
    for(int i = 0; i < size; i++)
    {
        dest[i] = src[i];
    }
    return true;
}

bool CopyCArrayToStdVector(const double* src, vector<double>& dest, int size)
{
    if(!src)
    {
        Log(lError)<<"Tried to copy from NULL vector";
        return false;
    }

    dest.resize(size);
    for(int i = 0; i < size; i++)
    {
        dest[i] = src[i];
    }
    return true;
}

bool CopyCArrayToStdVector(const bool* src, vector<bool>& dest, int size)
{
    if(!src)
    {
        Log(lError)<<"Tried to copy from NULL vector";
        return false;
    }

    dest.resize(size);
    for(int i = 0; i < size; i++)
    {
        dest[i] = src[i];
    }
    return true;
}

double* CreateVector(const vector<double>& vec)
{
    double* avec = new double[vec.size()];
    if(!avec)
    {
        Log(lError)<<"Failed to allocate c vector";
        return NULL;
    }

    for(int i = 0; i < vec.size(); i++)
    {
          avec[i] = vec[i];
    }
    return avec;
}


StringList getSelectionListFromSettings(const SimulationSettings& settings)
{
	//read from settings the variables found in the amounts and concentrations lists
	StringList theList;
	TSelectionRecord record;

    int nrOfVars = settings.mVariables.Count();

	for(int i = 0; i < settings.mAmount.Count(); i++)
	{
		theList.Add("[" + settings.mAmount[i] + "]");        //In the setSelection list below, the [] selects the correct 'type'
	}

	for(int i = 0; i < settings.mConcentration.Count(); i++)
	{
		theList.Add(settings.mConcentration[i]);
	}

    //We may have variables
    //A variable 'exists' only in "variables", not in the amount or concentration section
    int currCount = theList.Count();
	if( nrOfVars > currCount)
	{
        //Look for a variable that is not in the list

        for(int i = 0; i < settings.mVariables.Count(); i++)
		{
            string aVar = settings.mVariables[i];
            if(settings.mAmount.DontContain(aVar) && settings.mConcentration.DontContain(aVar))
            {
			    theList.Add(settings.mVariables[i]);
            }

        }
    }

  	theList.InsertAt(0, "time");
    return theList;
}

#if defined(_WIN32) || defined(WIN32)

string GetWINAPIError(DWORD errorCode, LPTSTR lpszFunction)
{
 	LPVOID lpMsgBuf;
    LPVOID lpDisplayBuf;
    DWORD dw = GetLastError();

    FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER |
        FORMAT_MESSAGE_FROM_SYSTEM |
        FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,
        dw,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPTSTR) &lpMsgBuf,
        0, NULL
    );

    // Display the error message
    lpDisplayBuf = (LPVOID)LocalAlloc(LMEM_ZEROINIT, (lstrlen((LPCTSTR)lpMsgBuf) + lstrlen((LPCTSTR)lpszFunction) + 40) * sizeof(TCHAR));

    StringCchPrintf((LPTSTR)lpDisplayBuf,
        				LocalSize(lpDisplayBuf) / sizeof(TCHAR),
        				TEXT("%s failed with error %d: %s"),
        				lpszFunction,
                        dw,
                        lpMsgBuf);

    string errorMsg = string((LPCTSTR)lpDisplayBuf);
    LocalFree(lpMsgBuf);
    LocalFree(lpDisplayBuf);
    return errorMsg;
}

#endif

}//end of namespace
