#ifndef TRANSACTIONTHREAD_H
#define TRANSACTIONTHREAD_H

#include <QtGui>

#include <list>
#include "transaction.h"

class TransactionThread : public QThread
{
public:
    TransactionThread();
    void run();
/*    void setTargetWidget(QWidget *widget);*/
    void setTargetObject(QObject *_object);
    QObject * getTargetObject();
	void setStopThread(bool *_stopThread);//TEMP	

    void addTransaction(Transaction *transact);
	void initTransactions();
	int sizeTransactions(){return transactions.size();};

    QImage image();

private:
//     QWidget *targetWidget;
    QObject *targetObject;
    QMutex mutex;
    std::list<Transaction *> transactions;
	bool *pstopThread;//TEMP
};

#endif
