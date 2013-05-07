
#if defined(_WIN32) || defined(__CODEGEARC__)
	#if defined(EXPORT_POCO)
		#define POCO_API_TEST __declspec(dllexport)
	#else
		#define POCO_API_TEST __declspec(dllimport)
	#endif
#else
#define POCO_API_TEST 
#define __stdcall
#endif

extern "C" void POCO_API_TEST __stdcall poco_hello();

