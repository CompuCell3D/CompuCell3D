//---------------------------------------------------------------------------
#include <vcl.h>
#pragma hdrstop

#pragma comment(lib, "rrc_api.lib")
#pragma comment(lib, "roadrunner.lib")

//---------------------------------------------------------------------------
USEFORM("Main.cpp", MainF);
//---------------------------------------------------------------------------
int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int)
{
	try
	{
		Application->Initialize();
		Application->MainFormOnTaskBar = true;
		Application->CreateForm(__classid(TMainF), &MainF);
		Application->Run();
	}
	catch (Exception &exception)
	{
		Application->ShowException(&exception);
	}
	catch (...)
	{
		try
		{
			throw Exception("");
		}
		catch (Exception &exception)
		{
			Application->ShowException(&exception);
		}
	}
	return 0;
}
//---------------------------------------------------------------------------
