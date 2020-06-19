from PyQt5 import QtCore
from PyQt5.QtCore import *
from PyQt5.QtNetwork import *
import requests
import time


class WebFetcherRequestsThread(QThread):
    gotWebContentSignal = QtCore.pyqtSignal(str, str)

    def __init__(self, parent):
        QThread.__init__(self, parent)
        self.parent = parent
        self.url_str = ''


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
        print('demo')
        # start timer that on expiration emits empty content
        print('inside run')

        # time.sleep(5)
        code, content = self.get_web_content()

        # print('fetch=', self.url_str)
        #
        # print('request r = ', r)
        #
        # content = r.content.decode('utf-8')
        #
        # self.timer.stop()
        #
        self.emit_response(content=content, url_str=self.url_str)
        print()

    # def emit_empty(self):
    #     print('emitting empty')
    #     self.stop()
    #
    #     self.emit_response(url_str=self.url_str, content='')

    def emit_response(self, content, url_str):
        self.gotWebContentSignal.emit(content, url_str, )


class WebFetcherRequests(QObject):
    '''
    This class fetches content of web page
    '''
    # signal emited once the web page content is fetched - first argument is a full content fo the web page,
    # second is url from which the content is fetched
    gotWebContentSignal = QtCore.pyqtSignal(str, str)

    def __init__(self, _parent=None):
        super(WebFetcherRequests, self).__init__()
        self.parent = _parent
        # self.user_agent = b'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'
        # self.network_manager = QNetworkAccessManager()

        # self.network_manager.finished.connect(self.reply_finished)
        self.url_str = ''
        self.timer = None
        self.delay_msec = 1000

    def fetch(self, url_str):
        '''
        initiates web page fetch process. self.networkmanager emits 'finished' signal once the content has been fetched
        or if e.g. network error has occurred. In the latter case the content of the web page is an empty QString
        :param url_str: web address
        :return: None
        '''

        self.web_fetcher_thread = WebFetcherRequestsThread(self)
        self.web_fetcher_thread.set_url_str(url_str=url_str)

        self.web_fetcher_thread.gotWebContentSignal.connect(self.process_received_content)
        print('inside fetch')
        self.web_fetcher_thread.start()
        # self.handle_web_timeout()

        #
        # self.url_str = url_str
        # print('fetch=', url_str)
        # r = requests.get(url_str)
        # print('request r = ', r)
        #
        # r.content.decode('utf-8')

        # config = QSslConfiguration.defaultConfiguration()
        # request = QNetworkRequest()
        # request.setRawHeader(b'User-Agent', self.user_agent)
        # # config.setProtocol(QSsl.TlsV1_2)
        # # request.setSslConfiguration(config)
        # request.setUrl(QUrl(url_str))
        #
        # self.network_manager.get(request)
        #
        # # self.network_manager.get(QNetworkRequest(QUrl(url_str)))

    # def handle_web_timeout(self):
    #     self.timer = QTimer(self)
    #     self.timer.timeout.connect(self.emit_empty)
    #     self.timer.start(self.delay_msec)
    #
    # def emit_empty(self):
    #     print('emitting empty')
    #     self.timer.stop()
    #     self.web_fetcher_thread.exit(0)
    #
    #
    #     self.process_received_content(content='', url_str=self.url_str,)


    def process_received_content(self, content: str, url_str: str):
        """

        :param content:
        :param url_str:
        :return:
        """

        self.gotWebContentSignal.emit(content, url_str)

    # def reply_finished(self, reply):
    #     '''
    #     slot for the 'finished' signal from self.network_manager. This slots emits another signal
    #     'gotWebContentSignal' that takes two arguments - content of the webpage (QString) and the url (QString) from
    #     which the page is requested
    #     :param reply: QString - is the content of the web page
    #     :return:
    #     '''
    #
    #     data = reply.readAll()
    #     data_qstring = str(data)
    #     data_qstring = str(data, 'utf-8')
    #     data_str = str(data_qstring)
    #     print('data_qstring=', data_qstring)
    #     self.gotWebContentSignal.emit(data_qstring, str(self.url_str))
    #
    # def run(self):
    #     self.fetch(self.url_str)


if __name__ == "__main__":
    from PyQt5.QtCore import QCoreApplication, QUrl
    import sys
    import time


    def print_response(content, url_str):
        print('got content=', content)
        print('URL=', url_str)


    app = QCoreApplication([])
    ex = WebFetcherRequests(app)
    ex.gotWebContentSignal.connect(print_response)
    ex.fetch(url_str="http://www.compucell3d.org/current_version")

    sys.exit(app.exec_())
