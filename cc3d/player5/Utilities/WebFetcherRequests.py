from PyQt5 import QtCore
from PyQt5.QtCore import *
from PyQt5.QtNetwork import *
import requests
import time
from weakref import ref


class WebFetcherRequestsThread(QThread):
    gotWebContentSignal = QtCore.pyqtSignal(str, str)

    def __init__(self, parent):
        QThread.__init__(self, parent)
        self.parent = parent
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

    def set_url_str(self, url_str: str):
        """
        sets url to fetch

        :param url_str:
        :return:
        """
        self.url_str = url_str

    def get_web_content(self):
        """
        returns web content

        :return:
        """
        ok = False
        content = ''
        try:
            r = requests.get(self.url_str)

        except requests.exceptions.ConnectionError:
            return ok, content
        except:
            return ok, content

        ok = r.ok

        if r.ok:
            content = r.content.decode(r.encoding)

        return ok, content

    def run(self):
        code, content = self.get_web_content()
        self.emit_response(content=content, url_str=self.url_str)

    def emit_response(self, content, url_str):
        self.gotWebContentSignal.emit(content, url_str, )


class WebFetcherRequests(QObject):
    """
    This class fetches content of web page
    """
    # signal emited once the web page content is fetched - first argument is a full content fo the web page,
    # second is url from which the content is fetched
    gotWebContentSignal = QtCore.pyqtSignal(str, str)

    def __init__(self, _parent=None):
        super(WebFetcherRequests, self).__init__()
        self.parent = _parent
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

        self.web_fetcher_thread = WebFetcherRequestsThread(self)
        self.web_fetcher_thread.set_url_str(url_str=url_str)

        self.web_fetcher_thread.gotWebContentSignal.connect(self.process_received_content)
        self.web_fetcher_thread.start()

    def process_received_content(self, content: str, url_str: str):
        """

        :param content:
        :param url_str:
        :return:
        """

        self.gotWebContentSignal.emit(content, url_str)


if __name__ == "__main__":
    from PyQt5.QtCore import QCoreApplication, QUrl
    import sys

    def print_response(content, url_str):
        print('got content=', content)
        print('URL=', url_str)


    app = QCoreApplication([])
    ex = WebFetcherRequests(app)
    ex.gotWebContentSignal.connect(print_response)
    ex.fetch(url_str="http://www.compucell3d.org/current_version")

    sys.exit(app.exec_())
