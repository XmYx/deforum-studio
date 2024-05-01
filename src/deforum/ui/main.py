import copy
import os
import sys
import json

import numba
import numpy as np
from PyQt6.QtGui import QImage, QAction
from PyQt6.QtWidgets import QApplication, QTabWidget, QWidget, QVBoxLayout, QDockWidget, QSlider, QLabel, QMdiArea, \
    QPushButton, QComboBox, QFileDialog, QSpinBox, QLineEdit, QCheckBox, QTextEdit
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot

from deforum.ui.core import DeforumCore
from deforum.ui.timeline import TimelineWidget
from deforum.utils.logging_config import logger

from PyQt6.QtGui import QPixmap
from PIL import ImageQt

class BackendThread(QThread):
    imageGenerated = pyqtSignal(object)  # Signal to emit the image data

    def __init__(self, params):
        super().__init__()
        self.params = params
        print(params)


    def run(self):
        try:
            from deforum.shared_storage import models

            # Load the deforum pipeline if not already loaded
            if "deforum_pipe" not in models:
                from deforum import DeforumAnimationPipeline
                models["deforum_pipe"] = DeforumAnimationPipeline.from_civitai(model_id="125703")
                                                                            #generator_name='DeforumDiffusersGenerator')
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
            self.params['enable_subseed_scheduling'] = True
            self.params['enable_steps_scheduling'] = True
            self.params['color_coherence'] = False
            animation = models["deforum_pipe"](callback=datacallback, **self.params) if not use_settings_file else models["deforum_pipe"](callback=datacallback, settings_file=file_path)
        except Exception as e:
            logger.info(repr(e))
def npArrayToQPixmap(arr):
    """Convert a numpy array to QPixmap."""
    height, width, _ = arr.shape
    bytes_per_line = 3 * width
    qImg = QImage(
        arr.data,
        width,
        height,
        bytes_per_line,
        QImage.Format.Format_RGB888,
    )
    # Convert QImage to QPixmap
    pixmap = QPixmap.fromImage(qImg)
    return pixmap
class MainWindow(DeforumCore):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deforum Animator")
        self.setupDynamicUI()
        self.currentTrack = None  # Store the currently selected track
        self.presets_folder = os.path.join(os.path.expanduser('~'), 'deforum', 'presets')
        self.loadPresetsDropdown()
        self.presetsDropdown.activated.connect(self.loadPresetsDropdown)

    def loadPresetsDropdown(self):
        current_text = self.presetsDropdown.currentText()  # Remember current text
        if not os.path.exists(self.presets_folder):
            os.makedirs(self.presets_folder)

        preset_files = [f for f in os.listdir(self.presets_folder) if f.endswith('.txt')]
        self.presetsDropdown.clear()
        self.presetsDropdown.addItems(preset_files)

        # Re-select the current text if it exists in the new list
        if current_text in preset_files:
            index = self.presetsDropdown.findText(current_text)
            self.presetsDropdown.setCurrentIndex(index)

    def loadPreset(self):
        selected_preset = self.presetsDropdown.currentText()
        preset_path = os.path.join(self.presets_folder, selected_preset)
        try:
            with open(preset_path, 'r') as file:
                config = json.load(file)
                for key, value in config.items():
                    if key in self.params:
                        self.params[key] = value
            self.updateUIFromParams()

        except:
            pass

    def savePreset(self):
        preset_name, _ = QFileDialog.getSaveFileName(self, 'Save Preset', self.presets_folder, 'Text Files (*.txt)')
        if preset_name:
            preset_name = os.path.splitext(os.path.basename(preset_name))[0] + '.txt'
            preset_path = os.path.join(self.presets_folder, preset_name)

            with open(preset_path, 'w') as file:
                json.dump(self.params, file, indent=4)

            self.loadPresetsDropdown()
    def setupDynamicUI(self):
        self.toolbar = self.addToolBar('Main Toolbar')
        self.renderButton = QAction('Render', self)
        self.stopRenderButton = QAction('Stop Render', self)

        self.toolbar.addAction(self.renderButton)
        self.toolbar.addAction(self.stopRenderButton)
        # Add presets dropdown to the toolbar
        self.presetsDropdown = QComboBox()
        self.presetsDropdown.currentIndexChanged.connect(self.loadPreset)
        self.toolbar.addWidget(self.presetsDropdown)

        # Add actions to load and save presets
        self.loadPresetAction = QAction('Load Preset', self)
        self.loadPresetAction.triggered.connect(self.loadPreset)
        self.toolbar.addAction(self.loadPresetAction)

        self.savePresetAction = QAction('Save Preset', self)
        self.savePresetAction.triggered.connect(self.savePreset)
        self.toolbar.addAction(self.savePresetAction)
        self.renderButton.triggered.connect(self.startBackendProcess)
        self.stopRenderButton.triggered.connect(self.stopBackendProcess)
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
                elif params['widget_type'] == 'float':
                    self.createDoubleSpinBox(params['label'], layout, params['min'], params['max'], 0.01, params['default'], setting)

                elif params['widget_type'] == 'dropdown':
                    self.createComboBox(params['label'], layout, [str(param) for param in params['options']], setting)
                elif params['widget_type'] == 'text input':
                    self.createTextBox(params['label'], layout, params['default'], setting)
                elif params['widget_type'] == 'text box':
                    self.createTextBox(params['label'], layout, params['default'], setting)
                elif params['widget_type'] == 'slider':
                    self.createSlider(params['label'], layout, params['min'], params['max'], params['default'], setting)
                elif params['widget_type'] == 'checkbox':
                    self.createCheckBox(params['label'], layout, bool(params['default']), setting)
                elif params['widget_type'] == 'file_input':
                    # Add file input handler if needed
                    pass

            tabWidget.addTab(tab, category)
        # self.createPushButton('Render', layout, self.startBackendProcess)

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
        #params = {key: widget.value() for key, widget in self.params.items() if hasattr(widget, 'value')}
        # if not self.thread:
        self.thread = BackendThread(self.params)
        self.thread.imageGenerated.connect(self.updateImage)
        self.thread.start()
    def stopBackendProcess(self):
        try:
            from deforum.shared_storage import models
            models["deforum_pipe"].gen.max_frames = len(models["deforum_pipe"].images)
        except:
            pass
    @pyqtSlot(dict)
    def updateImage(self, data):
        # Update the image on the label
        if 'image' in data:


            img = copy.deepcopy(data['image'])

            # qt_img = ImageQt.ImageQt(img.convert("RGB"))  # ImageQt object
            # qimage = QImage(qt_img)  # Convert to QImage
            qpixmap = npArrayToQPixmap(np.array(img).astype(np.uint8))  # Convert to QPixmap
            self.previewLabel.setPixmap(qpixmap)
            self.timelineWidget.add_image_to_track(qpixmap)
    def updateParam(self, key, value):
        super().updateParam(key, value)
        from deforum.shared_storage import models

        # Load the deforum pipeline if not already loaded
        if "deforum_pipe" in models:

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

            models["deforum_pipe"].live_update_from_kwargs(**self.params)
            print("UPDATED DEFORUM PARAMS")
    def updateUIFromParams(self):
        for widget in self.findChildren(QWidget):
            if hasattr(widget, 'accessibleName'):
                key = widget.accessibleName()
                if key in self.params:
                    value = self.params[key]
                    if isinstance(widget, QSpinBox):
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec())
