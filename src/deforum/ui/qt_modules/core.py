import json
import os

from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy.QtGui import QAction
from qtpy.QtWidgets import QMainWindow, QHBoxLayout, QSpinBox, QDoubleSpinBox, QCheckBox, \
    QComboBox, QFileDialog, QLineEdit, QSlider, \
    QDockWidget
from qtpy.QtWidgets import QWidget, QTextEdit, QLabel, QPushButton

from deforum import logger
from deforum.ui.qt_modules.custom_ui import CustomTextBox


class DeforumCore(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDockOptions(
            QMainWindow.DockOption.AllowNestedDocks |  # Allow docks to be nested inside each other
            QMainWindow.DockOption.AllowTabbedDocks    # Allow docking in tabs
        )
        # self.initMenu()
        self.params = {}
        self.widgets = {}

    def initMenu(self):
        self.menuBar = self.menuBar()
        self.fileMenu = self.menuBar.addMenu("&File")
        self.editMenu = self.menuBar.addMenu("&Edit")
        self.modulesMenu = self.menuBar.addMenu("&Modules")
        self.helpMenu = self.menuBar.addMenu("&Help")

    def addMenu(self, title):
        new_menu = self.menuBar.addMenu(title)
        return new_menu

    def addMenuItem(self, menu, title, function):
        action = QAction(title, self)
        action.triggered.connect(function)
        menu.addAction(action)
        return action

    def addSubMenu(self, menu, title):
        sub_menu = menu.addMenu(title)
        return sub_menu

    def addSubMenuItem(self, sub_menu, title, function):
        action = QAction(title, self)
        action.triggered.connect(function)
        sub_menu.addAction(action)
        return action

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.adjustDockWidth()

    def adjustDockWidth(self):
        # Set the dock width to be 30% of the main window width
        dockWidth = self.width() * 0.3
        self.controlsDock.setMaximumWidth(int(dockWidth))
        # Update the docked controls widget maximum width as well
        self.controlsDock.widget().setMaximumWidth(int(dockWidth))

    def toggleImageLabelFullscreen(self):
        if not self.imageLabelFullscreen:
            # Entering imageLabel fullscreen mode
            self.imageLabel.setParent(None)
            self.imageSubWindow.setWidget(None)
            self.imageLabel.setWindowFlags(Qt.WindowType.FramelessWindowHint)
            self.imageLabel.showFullScreen()
            self.imageLabel.installEventFilter(self.imageLabelEventFilter)  # Install the event filter
            self.imageLabelFullscreen = True
        else:
            # Exiting imageLabel fullscreen mode
            self.imageLabel.removeEventFilter(self.imageLabelEventFilter)  # Uninstall the event filter
            # Ensure the imageLabel is properly re-parented and displayed within the imageSubWindow
            self.imageLabel.setWindowFlags(Qt.WindowType.Widget)  # Reset window flags
            self.imageLabel.showNormal()  # Ensure it's not in fullscreen or maximized state
            self.imageSubWindow.setWidget(self.imageLabel)
            self.imageLabelFullscreen = False

    def createEventFilter(self):
        # Define an event filter
        class Filter(QtCore.QObject):
            parent = None

            def eventFilter(self, obj, event):
                if event.type() == QtCore.QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape:
                    self.parent.toggleImageLabelFullscreen()
                    return True
                elif event.type() == QtCore.QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Space:
                    self.parent.dumpFramesToVideo()
                return False
        filter = Filter(self)
        filter.parent = self
        return filter

    def addDock(self, title, area):
        dock = QDockWidget(title, self)
        dock.setObjectName(title)
        dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        # Assuming the main window's size is an appropriate reference for setting the dock's size
        main_width = self.size().width()
        dock.setMinimumSize(main_width // 2, 150)  # Set minimum size to half the main window's width
        self.addDockWidget(area, dock)
        return dock
    def createSlider(self, label, layout, minimum, maximum, value, key):
        # Creating a slider and adding it to the layout
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(minimum)
        slider.setMaximum(maximum)
        slider.setValue(value)
        slider.valueChanged.connect(lambda val, k=key: self.updateParam(k, val))
        self.params[key] = value
        layout.addWidget(QLabel(label))
        layout.addWidget(slider)

    def createSpinBox(self, label, layout, minimum, maximum, step, value, key):
        hbox = QHBoxLayout()
        spinBox = QSpinBox()
        spinBox.setAccessibleName(key)

        spinBox.setMinimum(minimum)
        spinBox.setMaximum(maximum)
        spinBox.setSingleStep(step)
        spinBox.setValue(value)
        spinBox.valueChanged.connect(lambda val, k=key: self.updateParam(k, val))
        self.params[key] = value
        hbox.addWidget(QLabel(label))
        hbox.addWidget(spinBox)
        layout.addLayout(hbox)
        self.widgets[key] = spinBox
        return spinBox

    def createDoubleSpinBox(self, label, layout, minimum, maximum, step, value, key):
        hbox = QHBoxLayout()
        spinBox = QDoubleSpinBox()
        spinBox.setAccessibleName(key)

        spinBox.setMinimum(minimum)
        spinBox.setMaximum(maximum)
        spinBox.setSingleStep(step)
        spinBox.setValue(value)
        spinBox.valueChanged.connect(lambda val, k=key: self.updateParam(k, val))
        self.params[key] = value
        hbox.addWidget(QLabel(label))
        hbox.addWidget(spinBox)
        layout.addLayout(hbox)
        self.widgets[key] = spinBox

        return spinBox

    def createTextBox(self, label, layout, value, key):
        hbox = QHBoxLayout()
        textBox = CustomTextBox(label, value)
        textBox.text_box.setText(value)
        textBox.text_box.setAccessibleName(key)
        textBox.setObjectName(key)
        # Connect the lambda directly without expecting arguments from the signal
        textBox.text_box.textChanged.connect(lambda: self.updateParam(key, textBox.text_box.toPlainText()))
        self.params[key] = value
        # hbox.addWidget(QLabel(label))
        hbox.addWidget(textBox)
        layout.addLayout(hbox)
        self.widgets[key] = textBox.text_box

        return textBox.text_box
    def createTextInput(self, label, layout, value, key):
        hbox = QHBoxLayout()
        textBox = QLineEdit()
        textBox.setText(value)
        textBox.setAccessibleName(key)
        # Connect the lambda directly without expecting arguments from the signal
        textBox.textChanged.connect(lambda: self.updateParam(key, textBox.text()))
        self.params[key] = value
        hbox.addWidget(QLabel(label))
        hbox.addWidget(textBox)
        layout.addLayout(hbox)
        self.widgets[key] = textBox

        return textBox
    def createCheckBox(self, label, layout, default, key):
        checkBox = QCheckBox(label)
        checkBox.setAccessibleName(key)

        checkBox.setChecked(bool(default))
        # checkBox.stateChanged.connect(lambda state, k=key: self.updateParam(k, state == checkBox.isChecked()))
        checkBox.stateChanged.connect(self.onStateChanged)
        layout.addWidget(checkBox)
        self.params[key] = default  # Initialize the parameter dictionary
        self.widgets[key] = checkBox

        return checkBox

    def createComboBox(self, label, layout, items, key):
        hbox = QHBoxLayout()
        comboBox = QComboBox()
        comboBox.setAccessibleName(key)

        comboBox.addItems(items)
        comboBox.currentTextChanged.connect(lambda val, k=key: self.updateParam(k, val))
        hbox.addWidget(QLabel(label))
        hbox.addWidget(comboBox)
        layout.addLayout(hbox)
        self.params[key] = comboBox.currentText()  # Initialize the parameter dictionary with the current text
        self.widgets[key] = comboBox
        return comboBox

    def createPushButton(self, label, layout, function):
        button = QPushButton(label)
        button.clicked.connect(function)
        layout.addWidget(button)
        return button

    # Method to handle value changes from SpinBox and DoubleSpinBox
    def onSpinBoxValueChanged(self, value):
        widget = self.sender()
        if isinstance(widget, QWidget):

            if widget:
                key = widget.accessibleName()
                self.updateParam(key, value)

    # Method to handle text changes from TextBox
    def onTextChanged(self):
        widget = self.sender()
        if isinstance(widget, QWidget):

            if widget:
                key = widget.accessibleName()
                self.updateParam(key, widget.toPlainText())

    # Method to handle state changes from CheckBox
    def onStateChanged(self, state):
        widget = self.sender()
        if isinstance(widget, QWidget):
            if widget:
                key = widget.accessibleName()
                self.updateParam(key, widget.isChecked())
    def updateParam(self, key, value):
        self.params[key] = value
    def populatePresetsDropdown(self):
        presetsPath = "presets"
        if not os.path.exists(presetsPath):
            os.makedirs(presetsPath)
        presets = [preset for preset in os.listdir(presetsPath) if preset.endswith('.json')]
        self.presetComboBox.clear()
        self.presetComboBox.addItems(presets)

    def savePreset(self):
        try:
            fileName, _ = QFileDialog.getSaveFileName(self, "Save Preset", "presets", "JSON Files (*.json)")
            if fileName:
                with open(fileName, 'w') as f:
                    json.dump(self.params, f, indent=4)
                self.populatePresetsDropdown()
        except Exception as e:
            logger.info("Failed to save preset:", str(e))

    def loadPreset(self, index):
        try:
            if index >= 0:  # To avoid loading during the clearing of the combo box
                fileName = self.presetComboBox.currentText()
                fullPath = os.path.join("presets", fileName)
                with open(fullPath, 'r') as f:
                    settings = json.load(f)
                for key, value in settings.items():
                    self.updateWidgetValue(key, value)
        except Exception as e:
            logger.info("Failed to load preset:", str(e))

    def updateWidgetValue(self, key, value):
        widget = self.widgets.get(key)
        if widget:
            if isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                widget.setValue(value)
            elif isinstance(widget, QCheckBox):
                widget.setChecked(value)
            elif isinstance(widget, QComboBox):
                widget.setCurrentIndex(widget.findText(value))
            elif isinstance(widget, QLineEdit) or isinstance(widget, QTextEdit):
                widget.setText(str(value))
            elif isinstance(widget, CustomTextBox):
                widget.text_box.setText(str(value))
            self.updateParam(key, value)

    def updateUIFromParams(self):
        for k, widget in self.widgets.items():
            if hasattr(widget, 'accessibleName'):
                key = widget.accessibleName()
                if key in self.params:
                    value = self.params[key]
                    if isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                        widget.setValue(value)
                    elif isinstance(widget, QComboBox):
                        index = widget.findText(str(value))
                        if index != -1:
                            widget.setCurrentIndex(index)
                    elif isinstance(widget, QLineEdit):
                        widget.setText(str(value))
                    elif isinstance(widget, QTextEdit):
                        widget.setText(str(value))
                    elif isinstance(widget, QSlider):
                        widget.setValue(value)
                    elif isinstance(widget, QCheckBox):
                        widget.setChecked(value)