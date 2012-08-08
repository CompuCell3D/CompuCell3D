#pragma once
#include <fstream>
#include <streambuf>
#include <locale>

#include "StreamRedirectorsDLLSpecifier.h"
//namespace std{

	class STREAMREDIRECTORS_EXPORT CustomStreamBufferBase: public std::streambuf{
	public:
		CustomStreamBufferBase(){}
		virtual ~CustomStreamBufferBase(){}
		virtual void setQTextEditPtr(void * _qTextEditPtr){}	
	protected:

	};

	//class STREAMREDIRECTORS_EXPORT CustomStreamBufferFactory{
	//public:
	//	CustomStreamBufferBase * getQTextEditBuffer();
	//};
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//};








class QTextEdit;
class QSender; //this is custom QObject class - not from Q
//namespace std{

	class STREAMREDIRECTORS_EXPORT QTextEditBuffer: public CustomStreamBufferBase{
	public:

		QTextEditBuffer();
		virtual void setQTextEditPtr(void * _qTextEditPtr);

		virtual ~QTextEditBuffer();
	protected:
		static const int bufferSize=16;
		char buffer[bufferSize];

		//this function actually writes the content of the internal buffer
		int flushBuffer();

		//this function will be called whenever buffer is about to get full - streambuf will keep appending characters to the buffer
		//but once it will see the internal buffer is about to get full it will call overflow function. For unbuffered streams overflow is called for each character
		virtual int_type overflow(int_type c);
		virtual int sync();
	private:
		QTextEdit * qTextEditPtr;
		QSender * qSender;

	};





//};