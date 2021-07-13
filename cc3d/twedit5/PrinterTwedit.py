from cc3d.twedit5.twedit.utils.global_imports import *
from PyQt5.QtPrintSupport import QPrinter


# this code is basedon Eric 4 code

class PrinterTwedit(QsciPrinter):
    """

    Class implementing the QextScintillaPrinter with a header.

    """

    def __init__(self, mode=QPrinter.ScreenResolution):
        """

        Constructor

        

        @param mode mode of the printer (QPrinter.PrinterMode)

        """

        QsciPrinter.__init__(self, mode)

        self.time = QTime.currentTime().toString(Qt.LocalDate)

        self.date = QDate.currentDate().toString(Qt.LocalDate)

    def formatPage(self, painter, drawing, area, pagenr):
        """

        Private method to generate a header line.

        

        @param painter the paint canvas (QPainter)

        @param drawing flag indicating that something should be drawn

        @param area the drawing area (QRect)

        @param pagenr the page number (int)

        """

        fn = self.docName()

        # formatting header

        header = QApplication.translate('Printer', '%1 - Printed on %2, %3 ').arg(fn).arg(self.date).arg(self.time)

        header.insert(0, "Twedit++5 : ")

        # formatting footer

        footer = QApplication.translate('Printer', 'Page %1').arg(pagenr)

        painter.save()

        painter.setFont(QFont("Arial"))  # set our header font

        painter.setPen(QColor("#848484"))  # set color

        if drawing:
            # printing header

            painter.drawText(area.left(), area.top() + painter.fontMetrics().ascent() + 5, header)

            # printing footer

            painter.drawText((area.width() - painter.fontMetrics().width(footer)) / 2, area.bottom() - 5, footer)

        # resetting main print area bounding rectangle   

        area.setTop(area.top() + painter.fontMetrics().height() + 10)

        area.setBottom(area.bottom() - (painter.fontMetrics().height() + 10))

        painter.restore()
