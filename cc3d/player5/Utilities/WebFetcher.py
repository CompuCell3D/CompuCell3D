from PyQt5 import QtCore
from PyQt5.QtCore import *
from PyQt5.QtNetwork import *
from weakref import ref


class WebFetcher(QObject):
    """
    This class fetches content of web page
    """
    # signal emited once the web page content is fetched - first argument is a full content fo the web page,
    # second is url from which the content is fetched
    gotWebContentSignal = QtCore.pyqtSignal(str, str)

    def __init__(self, _parent=None):
        super(WebFetcher, self).__init__()
        self.parent = _parent
        self.user_agent = b'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'
        self.network_manager = QNetworkAccessManager()

        self.network_manager.finished.connect(self.reply_finished)
        self.url_str = ''

    @property
    def parent(self):
        try:
            o = self._parent()
        except TypeError:
            o = self._parent
        return o

    @parent.setter
    def parent(self, _i):
        try:
            self._parent = ref(_i)
        except TypeError:
            self._parent = _i

    def fetch(self, url_str):
        """
        initiates web page fetch process. self.networkmanager emits 'finished' signal once the content has been fetched
        or if e.g. network error has occurred. In the latter case the content of the web page is an empty QString

        :param url_str: web address
        :return: None
        """
        self.url_str = url_str
        print('fetch=', url_str)

        request = QNetworkRequest()
        request.setRawHeader(b'User-Agent', self.user_agent)
        request.setUrl(QUrl(url_str))

        self.network_manager.get(request)

    def reply_finished(self, reply):
        """
        slot for the 'finished' signal from self.network_manager. This slots emits another signal
        'gotWebContentSignal' that takes two arguments - content of the webpage (QString) and the url (QString) from
        which the page is requested

        :param reply: QString - is the content of the web page
        :return:
        """

        data = reply.readAll()
        data_qstring = str(data)
        data_qstring = str(data, 'utf-8')
        data_str = str(data_qstring)
        print('data_qstring=', data_qstring)
        self.gotWebContentSignal.emit(data_qstring, str(self.url_str))


if __name__ == "__main__":
    from PyQt5.QtCore import QCoreApplication, QUrl
    import sys

    app = QCoreApplication([])
    ex = WebFetcher()
    ex.fetch(url_str="http://www.compucell3d.org/current_version")
    sys.exit(app.exec_())
