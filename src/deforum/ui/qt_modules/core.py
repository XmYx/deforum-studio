import json
import os
import threading

from PyQt6 import QtCore
from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QAction, QPalette
from PyQt6.QtWidgets import QMainWindow, QMessageBox, QHBoxLayout, QSpinBox, QLabel, QDoubleSpinBox, QCheckBox, \
    QComboBox, QPushButton, QFileDialog, QTextEdit, QWidget, QVBoxLayout, QApplication


class CustomTextBox(QWidget):
    def __init__(self, label, default):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.text_box = QTextEdit(self)
        self.text_box.setText(default)
        self.label = QLabel(label, self.text_box)
        self.setupUI()
        self.updateStyles()

    def setupUI(self):
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)  # Make label ignore mouse events
        self.layout.addWidget(self.text_box)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Update label size and position on resize
        self.label.setGeometry(QRect(0, 0, self.text_box.width(), self.text_box.height()))

    def updateLabelVisibility(self):
        # Hide label if there is text, otherwise show it
        if self.text_box.toPlainText():
            self.label.hide()
        else:
            self.label.show()

    def updateStyles(self):
        # Adjust style based on the system's theme
        theme = QApplication.instance().palette().color(QPalette.ColorRole.Window).lightness()
        if theme > 128:  # Light theme
            text_color = "black"
            bg_label_color = "rgba(160, 160, 160, 100)"  # Light gray with some transparency
        else:  # Dark theme
            text_color = "white"
            bg_label_color = "rgba(255, 255, 255, 100)"  # Light white with some transparency

        self.text_box.setStyleSheet(f"color: {text_color}; background-color: transparent;")
        self.label.setStyleSheet(f"color: {bg_label_color}; font-size: 24px; font-weight: bold;")

class DeforumCore(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initMenu()
        self.params = {}

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
        return spinBox

    def createTextBox(self, label, layout, value, key):
        hbox = QHBoxLayout()
        textBox = CustomTextBox(label, value)
        textBox.text_box.setText(value)
        textBox.text_box.setAccessibleName(key)
        # Connect the lambda directly without expecting arguments from the signal
        textBox.text_box.textChanged.connect(lambda: self.updateParam(key, textBox.text_box.toPlainText()))
        self.params[key] = value
        # hbox.addWidget(QLabel(label))
        hbox.addWidget(textBox)
        layout.addLayout(hbox)
        return textBox.text_box

    def createCheckBox(self, label, layout, default, key):
        checkBox = QCheckBox(label)
        checkBox.setAccessibleName(key)

        checkBox.setChecked(bool(default))
        # checkBox.stateChanged.connect(lambda state, k=key: self.updateParam(k, state == checkBox.isChecked()))
        checkBox.stateChanged.connect(self.onStateChanged)
        layout.addWidget(checkBox)
        self.params[key] = default  # Initialize the parameter dictionary
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
        return comboBox

    def createPushButton(self, label, layout, function):
        button = QPushButton(label)
        button.clicked.connect(function)
        layout.addWidget(button)
        return button

    # Method to handle value changes from SpinBox and DoubleSpinBox
    def onSpinBoxValueChanged(self, value):
        widget = self.sender()
        if widget:
            key = widget.accessibleName()
            self.updateParam(key, value)

    # Method to handle text changes from TextBox
    def onTextChanged(self):
        widget = self.sender()
        if widget:
            key = widget.accessibleName()
            self.updateParam(key, widget.toPlainText())

    # Method to handle state changes from CheckBox
    def onStateChanged(self, state):
        widget = self.sender()
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
            print("Failed to save preset:", str(e))

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
            print("Failed to load preset:", str(e))

    def updateWidgetValue(self, key, value):
        widget = getattr(self, key + "_widget", None)
        if widget:
            if isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                widget.setValue(value)
            elif isinstance(widget, QCheckBox):
                widget.setChecked(value)
            elif isinstance(widget, QComboBox):
                widget.setCurrentIndex(widget.findText(value))
            self.updateParam(key, value)

    def loadConfig(self, configPath):
        """Load configuration from a specified JSON file."""
        try:
            with open(configPath, 'r') as f:
                settings = json.load(f)
            # Apply settings to UI components
            self.width_widget.setValue(settings["width"])
            self.height_widget.setValue(settings["height"])
            self.blend_widget.setValue(settings["blend"])
            self.cond_blend_widget.setValue(settings["conditional_blend"])
            self.fps_widget.setValue(settings["fps"])
            self.seed_widget.setValue(settings["seed"])
            self.inter_widget.setValue(settings["interpolation"])
            self.useLLM.setChecked(settings["use_llm"])
            self.llmDropdown.setCurrentText(settings["llm_model"])
            self.flowDropdown.setCurrentText(settings["flow_algo"])
            self.useFlow.setChecked(settings["use_flow"])
        except:
            pass