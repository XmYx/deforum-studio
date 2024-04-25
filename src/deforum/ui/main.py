import os
import sys
import json

from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication, QTabWidget, QWidget, QVBoxLayout, QDockWidget, QSlider, QLabel, QMdiArea, \
    QPushButton
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from deforum.ui.core import DeforumCore
from deforum.ui.timeline import TimelineWidget


class BackendThread(QThread):
    imageGenerated = pyqtSignal(object)  # Signal to emit the image data

    def __init__(self, params):
        super().__init__()
        self.params = params


    def run(self):
        from deforum.shared_storage import models

        # Load the deforum pipeline if not already loaded
        if "deforum_pipe" not in models:
            from deforum import DeforumAnimationPipeline
            models["deforum_pipe"] = DeforumAnimationPipeline.from_civitai(model_id="125703",
                                                                           generator_name='DeforumDiffusersGenerator')

        prom = self.params.get('prompts', 'cat sushi')
        key = self.params.get('keyframes', '0')
        if prom == "":
            prom = "Abstract art"
        if key == "":
            key = "0"

        if not isinstance(prom, dict):
            new_prom = list(prom.split("\n"))
            new_key = list(key.split("\n"))
            self.params["animation_prompts"] = dict(zip(new_key, new_prom))
        else:
            self.params["animation_prompts"] = prom

        # Call the deforum animation pipeline
        def datacallback(data):
            self.imageGenerated.emit(data)  # Emit the image data when available

        use_settings_file = False
        if 'settings_file' in self.params:
            file_path = self.params.pop('settings_file')
            if file_path:
                use_settings_file = True

        animation = models["deforum_pipe"](callback=datacallback, **self.params) if not use_settings_file else models["deforum_pipe"](callback=datacallback, settings_file=file_path)


class MainWindow(DeforumCore):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deforum Animator")
        self.setupDynamicUI()
        self.currentTrack = None  # Store the currently selected track
    def setupDynamicUI(self):
        # Initialize the tabbed control layout docked on the left
        self.controlsDock = self.addDock("Controls", Qt.DockWidgetArea.LeftDockWidgetArea)
        tabWidget = QTabWidget()
        self.controlsDock.setWidget(tabWidget)

        # Load UI configuration from JSON
        curr_folder = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(curr_folder, 'ui.json')
        with open(json_path, 'r') as file:
            config = json.load(file)

        for category, settings in config.items():
            tab = QWidget()
            layout = QVBoxLayout()
            tab.setLayout(layout)

            for setting, params in settings.items():
                if params['widget_type'] == 'number':
                    self.createSpinBox(params['label'], layout, params['min'], params['max'], 1, params['default'], setting)
                elif params['widget_type'] == 'dropdown':
                    self.createComboBox(params['label'], layout, [str(param) for param in params['options']], setting)
                elif params['widget_type'] == 'text input':
                    self.createTextBox(params['label'], layout, params['default'], setting)
                elif params['widget_type'] == 'text box':
                    self.createTextBox(params['label'], layout, params['default'], setting)
                elif params['widget_type'] == 'slider':
                    self.createSlider(params['label'], layout, params['min'], params['max'], params['default'], setting)
                elif params['widget_type'] == 'file_input':
                    # Add file input handler if needed
                    pass

            tabWidget.addTab(tab, category)
        self.createPushButton('Render', layout, self.startBackendProcess)

        # Create preview area
        self.setupPreviewArea()

        # self.timeline = TimelineWidget()
        # self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.timeline)
        # Create the timeline dock widget
        self.timelineDock = QDockWidget("Timeline", self)
        self.timelineDock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)

        # Create and set the timeline widget inside the dock
        self.timelineWidget = TimelineWidget()
        self.timelineDock.setWidget(self.timelineWidget)

        # Add the dock widget to the main window
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.timelineDock)

    def setupPreviewArea(self):
        # Setting up an MDI area for previews
        self.mdiArea = QMdiArea()
        self.setCentralWidget(self.mdiArea)
        self.previewSubWindow = self.mdiArea.addSubWindow(QWidget())
        self.previewLabel = QLabel("No preview available")
        self.previewLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.previewSubWindow.setWidget(self.previewLabel)
        self.previewSubWindow.show()

    def addDock(self, title, area):
        dock = QDockWidget(title, self)
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

    def startBackendProcess(self):
        params = {key: widget.value() for key, widget in self.params.items() if hasattr(widget, 'value')}
        self.thread = BackendThread(params)
        self.thread.imageGenerated.connect(self.updateImage)
        self.thread.start()

    def updateImage(self, data):
        # Update the image on the label
        if 'image' in data:
            from PyQt6.QtGui import QPixmap
            from PIL import ImageQt
            qt_img = ImageQt.ImageQt(data['image'])  # ImageQt object
            qimage = QImage(qt_img)  # Convert to QImage
            qpixmap = QPixmap.fromImage(qimage)  # Convert to QPixmap
            self.previewLabel.setPixmap(qpixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec())
