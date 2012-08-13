#include "CustomStreamBuffers.h"
#include "QSender.h"

// #ifdef USE_STREAM_REDIRECTORS
#include <iostream>
#include <QtGui>
#include <QtCore>


using namespace std;




QTextEditBuffer::QTextEditBuffer():
qTextEditPtr(0),
qSender(0)
{
	setp(buffer,buffer+(bufferSize-1)); //initialize internal buffer for writing so that str4eambuf knows where to put consecutive characters
	qSender=new QSender;
}
QTextEditBuffer::~QTextEditBuffer(){
	//sync();
	if (qSender)
		delete qSender;
}

void QTextEditBuffer::setQTextEditPtr(void *_qTextEditPtr){
	qTextEditPtr=(QTextEdit *)_qTextEditPtr;
	
	qSender->connect(qSender,SIGNAL(sendString(const QString & )),SLOT(colorText(const QString& )));	
	qTextEditPtr->connect(qSender,SIGNAL(sendString(const QString & )),SLOT(insertPlainText(const QString& )));	
	
	qSender->qePtr=qTextEditPtr;
	qSender->connect(qSender,SIGNAL(sendString(const QString & )),SLOT(ensureCursorVisible(const QString& )));
	//qSender->connect(qSender,SIGNAL(sendString(const QString & )),SLOT(restoreOrigTextColor(const QString& )));	

	//qSender->connect(qSender,SIGNAL(sendString(const QString & )),SLOT(redirectText(const QString& )));	

	//qRegisterMetaType<QTextCursor>("QTextCursor&");

}

//this function actually writes the content of the internal buffer
int QTextEditBuffer::flushBuffer(){
	int num = pptr()-pbase();

	string str(buffer,num);
	QString qStr(str.c_str());
	////QString qStr(buffer,num);

	if (qTextEditPtr){		
		qSender->outputString(qStr);				
	}

	//if (qTextEditPtr){
	//	qTextEditPtr->insertPlainText(qStr);
	//}
	pbump(-num);
	return num;
}

////this function will be called whenever buffer is about to get full - streambuf will keep appending characters to the buffer
////but once it will see the internal buffer is about to get full it will call overflow function. For unbuffered streams overflow is called for each character



QTextEditBuffer::int_type QTextEditBuffer::overflow(int_type c){ 
	if (c!=EOF){
		*pptr()=c; //store character in the buffer
		pbump(1); //set pptr to the begining of the buffer 
	}
	if(flushBuffer()==EOF){
		//error
		return EOF;
	}
	return c;


}
int QTextEditBuffer::sync(){
	if(flushBuffer()==EOF){
		return -1;
	}	
	return 0;
}



//CustomStreamBufferBase * CustomStreamBufferFactory::getQTextEditBuffer(){
//	return new QTextEditBuffer;
//}
// #else

// CustomStreamBufferBase * CustomStreamBufferFactory::getQTextEditBuffer(){
	// return new CustomStreamBufferBase;
// }


// #endif


