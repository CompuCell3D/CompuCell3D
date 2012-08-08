#include "transactionthread.h"
#include <iostream>
using namespace std;

TransactionThread::TransactionThread(){
// iti    targetWidget=0;
    targetObject=0;
}

// void TransactionThread::setTargetWidget(QWidget *widget)
// {
//     targetWidget = widget;
//    cerr<<"TransactinThread.cpp targetWidget="<<targetWidget<<endl;
//    exit(0);
// 
// }

void TransactionThread::setTargetObject(QObject *_object){
   
   targetObject=_object;
//    cerr<<"TransactinThread.cpp targetObject="<<targetObject<<endl;
//    exit(0);

}
QObject * TransactionThread::getTargetObject(){return targetObject;}



void TransactionThread::addTransaction(Transaction *transact)
{
//	cerr << "void TransactionThread::addTransaction()    !!!!!!!!!!!!!!!!!! \n";
    QMutexLocker locker(&mutex); //releases mutex when the QMutexLocker is destructed
    transactions.push_back(transact);
    if (!isRunning())
        start();
}

void TransactionThread::initTransactions()
{
	transactions.clear();
}

void TransactionThread::run()
{
    Transaction *transact;
//	cerr << "TransactionThread::run() stopThread ADDRESS: " << pstopThread << "\n"; // TESTING THREADING

//	while(!*pstopThread)
//	{
        mutex.lock();
        if (transactions.empty()) {
            mutex.unlock();
			return;
            //break;
        }
        transact = *transactions.begin();
        transactions.pop_front();
        mutex.unlock();

		//cerr << "TransactionThread::run() !!!!!!!!!!!!!!!!!!!!!!!!!!!!  stopThread ADDRESS: " << pstopThread << "\n"; // TESTING THREADING

//         transact->setTargetWidget(targetWidget);
        transact->setTargetObject(targetObject);
        transact->applySimulation(pstopThread);//TEMP
//		cerr << "THREAD EXECUTION AFTER transact->applySimulation() !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";

        delete transact;

//        mutex.lock();//???
//       mutex.unlock();//???
//	}	

//	cerr << "THREAD STOPPED EXECUTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
}

void TransactionThread::setStopThread(bool *_stopThread)
{
	pstopThread = _stopThread;
}
/*TEMP*/

