import re

from qtpy.QtCore import Qt, Signal, QPoint, QRect
from qtpy.QtGui import QPalette
from qtpy.QtWidgets import QMdiSubWindow, QApplication, QTextEdit, QMenu
from qtpy.QtWidgets import QTabWidget, QWidget, QVBoxLayout, QDockWidget, QLabel, QPushButton, QSpinBox, QLineEdit, \
    QCheckBox, QHBoxLayout, QDoubleSpinBox, QScrollArea, QTabBar


class DetachableTabWidget(QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMovable(True)
        self.setTabBarAutoHide(False)
        self.tabBar().setMouseTracking(True)
        self.main_window = parent  # Assume parent is the main window
        self.buttons = []
        self.docks = []

    def addTabWithDetachButton(self, widget, title):
        super().addTab(widget, title)
        tab_index = self.indexOf(widget)
        detach_button = QPushButton("Detach")
        detach_button.setProperty('widget', widget)  # Store the widget directly
        detach_button.setProperty('title', title)  # Store the widget directly
        detach_button.clicked.connect(self.detach_as_dock)
        self.buttons.append(detach_button)
        self.tabBar().setTabButton(tab_index, QTabBar.ButtonPosition.RightSide, detach_button)

    def detach_as_dock(self):
        button = self.sender()
        if button:
            widget = button.property('widget')
            if widget is None:
                return

            index = self.indexOf(widget)  # Determine index at the time of click
            if index == -1:
                return  # This should not happen, but just in case

            scroll_area = QScrollArea()
            scroll_area.setWidget(widget)
            scroll_area.setWidgetResizable(True)

            title = button.property('title')
            dock = AutoReattachDockWidget(title, self.main_window, widget, self)
            widget.should_detach = True
            dock.setObjectName(title)
            dock.setWidget(scroll_area)
            dock.setFloating(True)
            dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
            dock.show()
            self.buttons.remove(button)
            self.docks.append(dock)

    def reattach_from_dock(self, dock_widget):
        # Reattach dock_widget to the tab widget
        if dock_widget and dock_widget.original_widget:
            dock_widget.original_widget.should_detach = False
            title = dock_widget.windowTitle()
            self.addTabWithDetachButton(dock_widget.original_widget, title)
            dock_widget.close()  # Close the dock widget


class AutoReattachDockWidget(QDockWidget):
    def __init__(self, title, parent=None, original_widget=None, tab_widget=None):
        super().__init__(title, parent)
        self.original_widget = original_widget
        self.tab_widget = tab_widget
        self.should_detach = False

    def closeEvent(self, event):
        if self.original_widget and self.tab_widget:
            scroll_area = self.widget()
            if isinstance(scroll_area, QScrollArea):
                widget = scroll_area.takeWidget()
                self.tab_widget.addTabWithDetachButton(widget, self.windowTitle())
        super().closeEvent(event)

class AspectRatioMdiSubWindow(QMdiSubWindow):
    def __init__(self, image_label, *args, **kwargs):
        super(AspectRatioMdiSubWindow, self).__init__(*args, **kwargs)
        self.image_label = image_label
        self.aspect_ratio = 1  # Default aspect ratio

        # Connect the image label's signal to update the aspect ratio
        self.image_label.pixmapChanged.connect(self.updateAspectRatio)

    def updateAspectRatio(self, new_aspect_ratio):
        self.aspect_ratio = new_aspect_ratio
        self.updateSize()

    def updateSize(self):
        if self.aspect_ratio > 0:
            new_width = self.width()
            new_height = int(new_width / self.aspect_ratio)
            self.resize(new_width, new_height)

    def resizeEvent(self, event):
        self.updateSize()  # Ensure the window size respects the new aspect ratio
        super(AspectRatioMdiSubWindow, self).resizeEvent(event)
class ResizableImageLabel(QLabel):
    pixmapChanged = Signal(float)  # Emit new aspect ratio

    def __init__(self, parent=None):
        super(ResizableImageLabel, self).__init__(parent)
        self.setScaledContents(True)
        self.pixmap_original = None

    def setPixmap(self, pixmap):
        self.pixmap_original = pixmap
        super().setPixmap(pixmap)
        self.updateMinimumSize()
        if pixmap and pixmap.width() and pixmap.height():
            aspect_ratio = pixmap.width() / pixmap.height()
            self.pixmapChanged.emit(aspect_ratio)  # Emit signal with new aspect ratio

    def updateMinimumSize(self):
        if self.pixmap_original:
            min_width = max(1, self.pixmap_original.width() * 0.1)
            min_height = max(1, self.pixmap_original.height() * 0.1)
            self.setMinimumSize(int(min_width), int(min_height))

    def resizeEvent(self, event):
        if self.pixmap_original:
            scaled_pixmap = self.pixmap_original.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            super().setPixmap(scaled_pixmap)
        super().resizeEvent(event)

class CustomTextBox(QWidget):
    def __init__(self, label, default):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.text_box = QTextEdit(self)
        self.text_box.setText(default)
        self.label = QLabel(label, self)
        self.plus_button = QPushButton('+', self)
        self.minus_button = QPushButton('-', self)
        self.scale = 0.1  # Default scale
        self.setupUI()
        # self.updateStyles()
        self.connectSignals()

    def setupUI(self):
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.layout.addWidget(self.text_box)
        self.plus_button.setFixedSize(20, 20)
        self.minus_button.setFixedSize(20, 20)
        self.updateTooltip()  # Initialize tooltips
        self.plus_button.hide()
        self.minus_button.hide()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.label.setGeometry(QRect(0, 0, self.text_box.width(), self.text_box.height()))
        self.updateButtonPositions()

    def updateButtonPositions(self):
        base_y = self.text_box.height() - 25
        self.plus_button.move(self.text_box.width() - 25, base_y - 20)
        self.minus_button.move(self.text_box.width() - 25, base_y)

    def connectSignals(self):
        self.text_box.textChanged.connect(self.checkTextAndUpdateButtons)
        self.plus_button.clicked.connect(lambda: self.adjustNumber(self.scale))
        self.minus_button.clicked.connect(lambda: self.adjustNumber(-self.scale))
        self.plus_button.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.minus_button.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.plus_button.customContextMenuRequested.connect(lambda: self.showContextMenu(self.plus_button))
        self.minus_button.customContextMenuRequested.connect(lambda: self.showContextMenu(self.minus_button))

    def showContextMenu(self, button):
        context_menu = QMenu(self)
        for scale in [100, 10, 1, 0.1, 0.01, 0.001]:
            action = context_menu.addAction(f'Scale: {scale}')
            action.triggered.connect(lambda _, s=scale: self.setScale(s))
        context_menu.exec(button.mapToGlobal(QPoint(0, 0)))

    def setScale(self, scale):
        self.scale = scale
        self.updateTooltip()

    def updateTooltip(self):
        tooltip_text = f"<b>Current Scale: {self.scale}</b>"
        self.plus_button.setToolTip(tooltip_text)
        self.minus_button.setToolTip(tooltip_text)

    def checkTextAndUpdateButtons(self):
        text = self.text_box.toPlainText()
        pattern = re.compile(r'\(-?\d+(\.\d+)?\)')
        if pattern.search(text):
            self.plus_button.show()
            self.minus_button.show()
        else:
            self.plus_button.hide()
            self.minus_button.hide()

    def adjustNumber(self, increment):
        text = self.text_box.toPlainText()
        pattern = re.compile(r'\((-?\d+(\.\d+)?)\)')
        matches = pattern.search(text)
        if matches:
            number = float(matches.group(1)) + increment
            new_text = pattern.sub(f'({number:.3f})', text)
            self.text_box.setPlainText(new_text)

    def updateStyles(self):
        theme = QApplication.instance().palette().color(QPalette.ColorRole.Window).lightness()
        if theme > 128:
            text_color = "black"
            bg_label_color = "rgba(160, 160, 160, 50)"
            tooltip_bg_color = "rgba(50, 50, 50, 200)"  # Dark background for tooltip
        else:
            text_color = "white"
            bg_label_color = "rgba(200, 200, 200, 50)"
            tooltip_bg_color = "rgba(50, 50, 50, 200)"

        self.text_box.setStyleSheet(f"color: {text_color}; background-color: transparent;")
        self.label.setStyleSheet(f"color: {bg_label_color}; font-size: 24px; font-weight: bold;")

        # Tooltip style for all tooltips in the application
        QApplication.instance().setStyleSheet(f"QToolTip {{ "
                                              f"color: {text_color}; "
                                              f"background-color: {tooltip_bg_color}; "
                                              f"border: 1px solid {text_color}; "
                                              f"font-weight: bold; "
                                              f"padding: 4px; }}")


class JobQueueItem(QWidget):
    def __init__(self, job_name, job_data, parent=None):
        super().__init__(parent)
        self.job_name = job_name
        self.job_data = job_data

        layout = QHBoxLayout(self)
        self.label = QLabel(job_name)
        self.status_label = QLabel("Batched")
        layout.addWidget(self.label)
        layout.addWidget(self.status_label)
        self.setLayout(layout)
    def markStatus(self, status):
        self.status_label.setText(str(status))

    def markRunning(self):
        self.status_label.setText("Running")
        self.job_data['is_running'] = True

    def markComplete(self):
        self.status_label.setText("Completed")
        self.job_data['is_running'] = False



class JobDetailPopup(QWidget):
    def __init__(self, job_data, parent=None):
        super(JobDetailPopup, self).__init__(parent)
        self.job_data = job_data
        self.setupUI()

    def setupUI(self):
        layout = QVBoxLayout(self)
        self.setLayout(layout)

        for key, value in self.job_data.items():
            if isinstance(value, bool):
                widget = QCheckBox(key)
                widget.setChecked(value)
            elif isinstance(value, int):
                widget = QSpinBox()
                widget.setValue(value)
            elif isinstance(value, float):
                widget = QDoubleSpinBox()
                widget.setValue(value)
            else:
                widget = QLineEdit(str(value))

            layout.addWidget(widget)

        # Adding Save and Cancel buttons
        btnSave = QPushButton("Save")
        btnCancel = QPushButton("Cancel")
        btnSave.clicked.connect(self.saveChanges)
        btnCancel.clicked.connect(self.close)

        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(btnSave)
        buttonLayout.addWidget(btnCancel)
        layout.addLayout(buttonLayout)

    def saveChanges(self):
        # Implement logic to update job data
        self.close()
