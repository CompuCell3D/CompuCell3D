from PyQt5 import QtCore, QtGui, QtOpenGL, QtWidgets


class SliderWithValue(QtWidgets.QSlider):

    def __init__(self, parent=None):
        super(SliderWithValue, self).__init__(parent)
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
        super(SliderWithValue, self).setMinimum(min_val * self.precision_factor)

    def setMaximum(self, max_val):
        super(SliderWithValue, self).setMaximum(max_val * self.precision_factor)

    def setValue(self, val):
        super(SliderWithValue, self).setValue(val * self.precision_factor)

    def setTickInterval(self, tick_interval):
        super(SliderWithValue, self).setTickInterval(tick_interval * self.precision_factor)

    def setSingleStep(self, step):
        super(SliderWithValue, self).setSingleStep(int(step * self.precision_factor))

    def setPageStep(self, step):
        super(SliderWithValue, self).setPageStep(int(step * self.precision_factor))

    def value(self):
        return super(SliderWithValue, self).value() / self.precision_factor

    # sliderWithValue.setTickPosition(QtWidgets.QSlider.TicksBelow)

    def paintEvent(self, event):
        QtWidgets.QSlider.paintEvent(self, event)

        # orig
        # curr_value = str(self.value() / 100.0)

        # curr_value = str(self.value() / self.precision_factor)
        print 'self.value()=', self.value()
        curr_value = str(self.value())
        curr_value = self.value()


        print 'precision_factor=', self.precision_factor
        # round_value = round(float(curr_value), self.decimal_precision)
        round_value = round(curr_value, self.decimal_precision)
        # curr_value_str = str(round_value)

        print ('round_value=', round_value)

        # # orig
        # curr_value = str(self.value() / 100.00)
        # # curr_value = str(self.value() / self.precision_factor)
        # print 'precision_factor=', self.precision_factor
        # round_value = round(float(curr_value), 2)
        # print ('round_value=',round_value)

        # # orig
        # # curr_value = str(self.value() / 100.0)
        # curr_value = str(self.value() / self.precision_factor)
        # print 'precision_factor=', self.precision_factor
        # round_value = round(float(curr_value), self.decimal_precision)
        # print ('round_value=',round_value)

        # curr_value = str(self.value())
        # print 'self.value=',self.value()
        # round_value = round(float(curr_value))
        # print 'round_value=',round_value

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


# if __name__ == '__main__':
#     app = QtWidgets.QApplication([])
#
#     win = QtWidgets.QWidget()
#     win.setWindowTitle('Test Slider with Text')
#     win.setMinimumSize(600, 400)
#     layout = QtWidgets.QVBoxLayout()
#     win.setLayout(layout)
#
#     decimal_precision = 2
#     precision_factor = 10**decimal_precision
#     current_value = 2.0
#
#     sliderWithValue = SliderWithValue(QtCore.Qt.Horizontal)
#     sliderWithValue.set_decimal_precision(decimal_precision)
#
#     sliderWithValue.setMinimum(0.0)
#     sliderWithValue.setMaximum(10 * current_value)
#
#     sliderWithValue.setTickInterval(precision_factor)
#     sliderWithValue.setSingleStep(int(0.5*precision_factor))
#     sliderWithValue.setPageStep(precision_factor)
#
#
#     # sliderWithValue.setTickInterval(1)
#     # sliderWithValue.setSingleStep(0.5)
#     # sliderWithValue.setPageStep(1)
#     sliderWithValue.setTickPosition(QtWidgets.QSlider.TicksBelow)
#     sliderWithValue.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
#     sliderWithValue.setValue(current_value)
#
#     layout.addWidget(sliderWithValue)
#
#     win.show()
#     app.exec_()

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

    sliderWithValue = SliderWithValue(QtCore.Qt.Horizontal)
    sliderWithValue.set_decimal_precision(decimal_precision)
    sliderWithValue.setTickInterval(1)
    sliderWithValue.setSingleStep(0.5)
    sliderWithValue.setPageStep(1)

    sliderWithValue.setMinimum(-5.0)
    sliderWithValue.setMaximum(10 * current_value)
    sliderWithValue.setValue(current_value)

    # sliderWithValue.setMinimum(0.0)
    # sliderWithValue.setMaximum(10*current_value*precision_factor)
    # sliderWithValue.setTickInterval(precision_factor)
    # sliderWithValue.setSingleStep(int(0.5*precision_factor))
    # sliderWithValue.setPageStep(precision_factor)
    sliderWithValue.setTickPosition(QtWidgets.QSlider.TicksBelow)
    sliderWithValue.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
    # sliderWithValue.setValue(current_value * precision_factor)

    layout.addWidget(sliderWithValue)

    win.show()
    app.exec_()
