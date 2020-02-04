from PyQt5 import QtNetwork
from PyQt5.QtCore import QCoreApplication, QUrl
import sys

from cc3d.player5.Utilities.WebFetcher import WebFetcher

class Example(WebFetcher):

    def __init__(self):
        super(Example, self).__init__()

        # self.doRequest()

    # def doRequest(self):
    #
    #     url = "http://www.compucell3d.org/current_version"
    #     req = QtNetwork.QNetworkRequest(QUrl(url))
    #
    #     self.nam = QtNetwork.QNetworkAccessManager()
    #     self.nam.finished.connect(self.handleResponse)
    #     self.nam.get(req)
    #
    # def handleResponse(self, reply):
    #
    #     er = reply.error()
    #
    #     if er == QtNetwork.QNetworkReply.NoError:
    #
    #         bytes_string = reply.readAll()
    #         print(str(bytes_string, 'utf-8'))
    #
    #     else:
    #         print("Error occured: ", er)
    #         print(reply.errorString())
    #
    #     QCoreApplication.quit()



# class Example:
#
#     def __init__(self):
#
#         self.doRequest()
#
#     def doRequest(self):
#
#         url = "http://www.compucell3d.org/current_version"
#         req = QtNetwork.QNetworkRequest(QUrl(url))
#
#         self.nam = QtNetwork.QNetworkAccessManager()
#         self.nam.finished.connect(self.handleResponse)
#         self.nam.get(req)
#
#     def handleResponse(self, reply):
#
#         er = reply.error()
#
#         if er == QtNetwork.QNetworkReply.NoError:
#
#             bytes_string = reply.readAll()
#             print(str(bytes_string, 'utf-8'))
#
#         else:
#             print("Error occured: ", er)
#             print(reply.errorString())
#
#         QCoreApplication.quit()
#

app = QCoreApplication([])
ex = Example()
ex.fetch(url_str="http://www.compucell3d.org/current_version")
sys.exit(app.exec_())