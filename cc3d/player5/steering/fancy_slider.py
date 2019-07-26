from PyQt5 import QtCore, QtGui, QtOpenGL, QtWidgets


class FancySlider(QtWidgets.QSlider):

    def __init__(self, parent=None):
        super(FancySlider, self).__init__(parent)
        self.decimal_precision = 3

        self.stylesheet = """
        QSlider::groove:vertical {
                background-color: #222;
                width: 30px;
        }

        QSlider::handle:vertical {
            border: 1px #438f99;
            border-style: outset;
            margin: -2px 0;
            width: 30px;
            height: 3px;

            background-color: #438f99;
        }

        QSlider::sub-page:vertical {
            background: #4B4B4B;
        }

        QSlider::groove:horizontal {
                background-color: #222;
                height: 30px;
        }

        QSlider::handle:horizontal {
            border: 1px #438f99;
            border-style: outset;
            margin: -2px 0;
            width: 30   px;
            height: 30px;

            background-color: #438f99;
        }

        QSlider::sub-page:horizontal {
            background: #4B4B4B;
        }
        """

        self.setStyleSheet(self.stylesheet)

    @property
    def precision_factor(self):
        return float(10 ** self.decimal_precision)

    def set_decimal_precision(self, decimal_precision):
        """
        Sets decimal precision of the slider
        :param decimal_precision:{int} decimal precision of the slider
        :return: None
        """
        assert (decimal_precision >= 0.0) and (decimal_precision <= 5.0), 'Decimal precision must be between 0 and 5'
        self.decimal_precision = int(round(decimal_precision))

    def setMinimum(self, min_val):
        super(FancySlider, self).setMinimum(min_val * self.precision_factor)

    def setMaximum(self, max_val):
        super(FancySlider, self).setMaximum(max_val * self.precision_factor)

    def setValue(self, val):
        super(FancySlider, self).setValue(val * self.precision_factor)

    def setTickInterval(self, tick_interval):
        super(FancySlider, self).setTickInterval(tick_interval * self.precision_factor)

    def setSingleStep(self, step):
        super(FancySlider, self).setSingleStep(int(step * self.precision_factor))

    def setPageStep(self, step):
        super(FancySlider, self).setPageStep(int(step * self.precision_factor))

    def value(self):
        return super(FancySlider, self).value() / self.precision_factor

    def minimum(self):
        return super(FancySlider, self).minimum() / self.precision_factor

    def maximum(self):
        return super(FancySlider, self).maximum() / self.precision_factor


    def set_default_behavior(self):
        """
        After setting min, max it sets "common sense" behavior of the slider
        :return: None
        """
        min_, max_ = self.minimum(), self.maximum()

        self.setTickInterval((max_ - min_)/50.)
        self.setSingleStep((max_ - min_)/100.)
        self.setPageStep((max_ - min_)/50.)

    def paintEvent(self, event):
        QtWidgets.QSlider.paintEvent(self, event)

        curr_value = self.value()
        round_value = round(curr_value, self.decimal_precision)

        # handling integers
        if not self.decimal_precision:
            round_value = int(round_value)

        painter = QtGui.QPainter(self)
        painter.setPen(QtGui.QPen(QtCore.Qt.white))

        font_metrics = QtGui.QFontMetrics(self.font())
        font_width = font_metrics.boundingRect(str(round_value)).width()
        font_height = font_metrics.boundingRect(str(round_value)).height()

        rect = self.geometry()
        if self.orientation() == QtCore.Qt.Horizontal:
            horizontal_x_pos = rect.width() - font_width - 5
            horizontal_y_pos = rect.height() * 0.75

            painter.drawText(QtCore.QPoint(horizontal_x_pos, horizontal_y_pos), str(round_value))

        elif self.orientation() == QtCore.Qt.Vertical:
            vertical_x_pos = rect.width() - font_width - 5
            vertical_y_pos = rect.height() * 0.75

            painter.drawText(QtCore.QPoint(rect.width() / 2.0 - font_width / 2.0, rect.height() - 5), str(round_value))
        else:
            pass

        painter.drawRect(rect)

if __name__ == '__main__':
    app = QtWidgets.QApplication([])

    win = QtWidgets.QWidget()
    win.setWindowTitle('Test Slider with Text')
    win.setMinimumSize(600, 400)
    layout = QtWidgets.QVBoxLayout()
    win.setLayout(layout)

    decimal_precision = 2
    precision_factor = 10 ** decimal_precision
    current_value = 2.0

    sliderWithValue = FancySlider(QtCore.Qt.Horizontal)
    sliderWithValue.set_decimal_precision(decimal_precision)
    # sliderWithValue.setTickInterval(1)
    # sliderWithValue.setSingleStep(0.5)
    # sliderWithValue.setPageStep(1)

    sliderWithValue.setMinimum(-5.0)
    sliderWithValue.setMaximum(10 * current_value)
    sliderWithValue.setValue(current_value)
    sliderWithValue.set_default_behavior()

    sliderWithValue.setTickPosition(QtWidgets.QSlider.TicksBelow)
    sliderWithValue.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)

    layout.addWidget(sliderWithValue)

    win.show()
    app.exec_()
