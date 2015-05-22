__author__ = 'm'

from PyQt4 import QtCore
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4.QtNetwork import *

class WebFetcher(QObject):
    '''
    This class fetches content of web page
    '''
    # signal emited once the web page content is fetched - first argument is a full content fo the web page,
    # second is url from which the content is fetched
    gotWebContentSignal = QtCore.pyqtSignal((QtCore.QString, QtCore.QString))

    def __init__(self, _parent=None):
        super(WebFetcher, self).__init__()
        # QObject.__init__(_parent)

        self.parent = _parent

        self.network_manager = QNetworkAccessManager()

        self.network_manager.finished.connect(self.reply_finished)
        self.url_str = ''
        # self.connect(self.network_manager, SIGNAL('finished'), self.reply_finished )

         # connect(m_manager, SIGNAL(finished(QNetworkReply*)),
         # this, SLOT(replyFinished(QNetworkReply*)));

    def fetch(self, url_str):
        '''
        initiates web page fetch process. self.networkmanager emits 'finished' signal once the content has been fetched
        or if e.g. network error has occured. In the latter case the content of the web page is an empty QString
        :param url_str: web address
        :return: None
        '''
        self.url_str = url_str
        print 'fetch=',url_str
        self.network_manager.get(QNetworkRequest(QUrl(url_str)))


    def reply_finished(self, reply):
        '''
        slot for the 'finished' signal from self.network_manager. This slots emits another signal
        'gotWebContentSignal' that takes two arguments - content of the webpage (QString) and the url (QString) from
        which the page is requested
        :param reply: QString - is the content of the web page
        :return:
        '''
        data = reply.readAll()
        data_qstring = QString(data)
        data_str = str(data_qstring)
        # print data_str
        self.gotWebContentSignal.emit(data_qstring, QString(self.url_str))



if __name__ == "__main__":
    import re

    str_http = '<span class="anchor" id="line-1"></span><p class="line874">current version 3.7.4 20150515 <span class="anchor" id="line-2"><'

    current_version_regex = re.compile("(current version)([0-9\. ]*)")

    # current_version_regex = re.compile(r'span')
    match_obj = re.search(current_version_regex, str_http)
    print match_obj.groups()


    str_http = 'line874">what is new: Improved Player, Simplified Python Scripting, Easier Specification Of Plots <span cla><this is '
    whats_new_regex = re.compile("(>[\S]*what is new:)(.*?)(<)")

    search_obj = re.search(whats_new_regex, str_http)
    print search_obj.groups()

