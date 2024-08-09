from tkinter import BaseWidget
from qtpy import QtWidgets as QtW, QtCore
from qtpy.QtCore import Qt, QEvent, QSize
from magicgui.application import use_app
from magicgui.widgets import Widget
from magicgui.widgets._concrete import _LabeledWidget
from magicgui.backends._qtpy.widgets import (
    QBaseWidget,
    Container as ContainerBase,
    MainWindow as MainWindowBase,
)

class _Disableable(ContainerBase):
    def __init__(self, layout="vertical", scrollable: bool = False, **kwargs):
        BaseWidget.__init__(self, QtW.QWidget)

        if layout == "horizontal":
            self._layout: QtW.QLayout = QtW.QHBoxLayout()
        else:
            self._layout = QtW.QVBoxLayout()

        self._stacked_widget = QtW.QStackedWidget(self._qwidget)
        self._stacked_widget.setContentsMargins(0, 0, 0, 0)
        self._inner_qwidget = QtW.QWidget(self._qwidget)
        self._qwidget.setLayout(self._layout)
        self._layout.addWidget(self._stacked_widget)
        self._layout.addWidget(self._inner_qwidget)

    def _mgui_insert_widget(self, position: int, widget: Widget):
        self._stacked_widget.insertWidget(position, widget.native)

    def _mgui_remove_widget(self, widget: Widget):
        self._stacked_widget.removeWidget(widget.native)
        widget.native.setParent(None)
