import copy
import os
import sys
import json

import numpy as np

from PyQt6.QtGui import QImage, QAction
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import QApplication, QTabWidget, QWidget, QVBoxLayout, QDockWidget, QSlider, QLabel, QMdiArea, \
    QPushButton, QComboBox, QFileDialog, QSpinBox, QLineEdit, QCheckBox, QTextEdit, QHBoxLayout, QListWidget, \
    QDoubleSpinBox, QListWidgetItem
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QUrl

from deforum.ui.qt_modules.core import DeforumCore
from deforum.ui.qt_modules.timeline import TimelineWidget
from deforum.utils.logging_config import logger

from PyQt6.QtGui import QPixmap


class BackendThread(QThread):
    imageGenerated = pyqtSignal(object)  # Signal to emit the image data
    finished = pyqtSignal(str)  # Signal to emit the image data

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
            models["deforum_pipe"].generator.optimize = self.params["optimize"]
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
            self.params['hybrid_use_first_frame_as_init_image'] = False
            animation = models["deforum_pipe"](callback=datacallback, **self.params) if not use_settings_file else models["deforum_pipe"](callback=datacallback, settings_file=file_path)
            if hasattr(animation, 'video_path'):
                self.finished.emit(animation.video_path)
        except Exception as e:
            logger.info(repr(e))
            self.finished.emit("Error")
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


class ResizableImageLabel(QLabel):
    def __init__(self, parent=None):
        # Replacing deprecated sip function with recommended alternative
        # Assuming you identified where sipPyTypeDict() was used and replaced it
        super(ResizableImageLabel, self).__init__(parent)
        self.setScaledContents(True)  # Enable scaling to allow both enlarging and reducing the pixmap size
        self.pixmap_original = None

    def setPixmap(self, pixmap):
        self.pixmap_original = pixmap  # Store the original pixmap
        super().setPixmap(pixmap)  # Set the initial pixmap
        self.setMinimumSize(0, 0)
    def resizeEvent(self, event):
        if self.pixmap_original:
            # Scale pixmap to fit the current label size while maintaining the aspect ratio
            scaled_pixmap = self.pixmap_original.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            super().setPixmap(scaled_pixmap)
            # Adjust minimum size based on a minimum scale factor, e.g., allow scaling down to 10% of original
            min_width = max(1, self.pixmap_original.width() * 0.1)
            min_height = max(1, self.pixmap_original.height() * 0.1)
            self.setMinimumSize(int(min_width), int(min_height))
        super().resizeEvent(event)

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


class MainWindow(DeforumCore):
    def __init__(self):
        super().__init__()
        self.player = QMediaPlayer()  # Create a media player object
        self.setWindowTitle("Deforum Engine")
        self.setupDynamicUI()
        self.currentTrack = None  # Store the currently selected track
        self.presets_folder = os.path.join(os.path.expanduser('~'), 'deforum', 'presets')
        self.loadPresetsDropdown()
        self.presetsDropdown.activated.connect(self.loadPresetsDropdown)
        self.setupVideoPlayer()
        self.tileMdiSubWindows()
        self.timelineDock.hide()
        self.setupBatchControls()
        self.job_queue = []  # List to keep jobs to be processed
        self.current_job = None  # To keep track of the currently running job

    def onJobDoubleClicked(self, item):
        widget = self.jobQueueList.itemWidget(item)
        if widget:
            # Create a JobDetailPopup and add it to the MDI area
            popup = JobDetailPopup(widget.job_data)
            subWindow = self.mdiArea.addSubWindow(popup)  # Add popup to MDI area
            popup.setWindowTitle("Job Details - " + widget.job_name)
            subWindow.show()

    def setupBatchControls(self):
        self.batchControlTab = QWidget()
        layout = QVBoxLayout()
        self.batchControlTab.setLayout(layout)
        self.jobQueueList = QListWidget()
        self.jobQueueList.itemDoubleClicked.connect(self.onJobDoubleClicked)

        # Buttons for batch control
        startButton = QPushButton("Start Batch")
        pauseButton = QPushButton("Pause Batch")

        # Connect buttons to their respective slots (functions)
        startButton.clicked.connect(self.startBatchProcess)
        pauseButton.clicked.connect(self.stopBatchProcess)

        layout.addWidget(startButton)
        layout.addWidget(pauseButton)
        layout.addWidget(self.jobQueueList)

        # Add the batch control tab to the main tab widget
        self.tabWidget.addTab(self.batchControlTab, "Batch Control")
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

    def addCurrentParamsAsJob(self):
        # Adding current params as a new job
        job_description = json.dumps(self.params, indent=4)
        item = QListWidgetItem(self.jobQueueList)
        widget = JobQueueItem("Job Name", self.params)
        item.setSizeHint(widget.sizeHint())
        self.jobQueueList.addItem(item)
        self.jobQueueList.setItemWidget(item, widget)
        self.job_queue.append(widget)  # Append job widget to the queue


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
        self.tileSubWindowsAction = QAction('Tile Subwindows', self)
        self.tileSubWindowsAction.triggered.connect(self.tileMdiSubWindows)
        self.toolbar.addAction(self.tileSubWindowsAction)
        self.savePresetAction = QAction('Save Preset', self)
        self.savePresetAction.triggered.connect(self.savePreset)
        self.toolbar.addAction(self.savePresetAction)

        self.addJobButton = QAction('Add to Batch', self)
        self.addJobButton.triggered.connect(self.addCurrentParamsAsJob)
        self.toolbar.addAction(self.addJobButton)

        self.renderButton.triggered.connect(self.startBackendProcess)
        self.stopRenderButton.triggered.connect(self.stopBackendProcess)
        # Initialize the tabbed control layout docked on the left
        self.controlsDock = self.addDock("Controls", Qt.DockWidgetArea.LeftDockWidgetArea)
        self.tabWidget = QTabWidget()
        self.controlsDock.setWidget(self.tabWidget)

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

            self.tabWidget.addTab(tab, category)
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
        self.previewLabel = ResizableImageLabel()
        self.previewLabel.setScaledContents(True)
        self.previewLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.previewSubWindow.setWidget(self.previewLabel)
        self.previewSubWindow.show()

        # # Subwindow for the video player
        # self.videoSubWindow = self.mdiArea.addSubWindow(QWidget())
        # self.videoSubWindow.setWindowTitle("Video Player")
        #
        # # Video output widget
        # self.videoWidget = QVideoWidget()
        #
        # # Video controls
        # self.playButton = QPushButton("Play")
        # self.pauseButton = QPushButton("Pause")
        # self.stopButton = QPushButton("Stop")
        #
        # # Connect buttons to player actions
        # self.playButton.clicked.connect(self.player.play)
        # self.pauseButton.clicked.connect(self.player.pause)
        # self.stopButton.clicked.connect(self.player.stop)
        #
        # # Slider for video timeline

        #
        # # Layout for the video player controls
        # controlsLayout = QHBoxLayout()
        # controlsLayout.addWidget(self.playButton)
        # controlsLayout.addWidget(self.pauseButton)
        # controlsLayout.addWidget(self.stopButton)
        #
        # # Main layout for the video subwindow
        # mainLayout = QVBoxLayout()
        # mainLayout.addWidget(self.videoWidget)
        # mainLayout.addWidget(self.videoSlider)
        # mainLayout.addLayout(controlsLayout)
        #
        # self.videoSubWindow.widget().setLayout(mainLayout)
        # self.player.setVideoOutput(self.videoWidget)
        # self.videoSubWindow.show()
    def setupVideoPlayer(self):
        self.videoSubWindow = self.mdiArea.addSubWindow(QWidget())
        self.videoSubWindow.setWindowTitle("Video Player")
        # Setting up the video player within its subwindow
        self.videoWidget = QVideoWidget()
        self.audioOutput = QAudioOutput()  # Create an audio output
        self.player.setAudioOutput(self.audioOutput)  # Set audio output for the player
        self.player.setVideoOutput(self.videoWidget)  # Set video output

        # Adding playback controls
        self.playButton = QPushButton("Play")
        self.pauseButton = QPushButton("Pause")
        self.stopButton = QPushButton("Stop")
        self.volumeSlider = QSlider(Qt.Orientation.Horizontal)
        self.volumeSlider.setMinimum(0)
        self.volumeSlider.setMaximum(100)
        self.volumeSlider.setValue(50)  # Default volume level at 50%
        self.volumeSlider.setTickInterval(10)  # Optional: makes it easier to slide in intervals
        self.audioOutput.setVolume(0.5)  # Set the initial volume to 50%

        # Connect control buttons and slider
        self.playButton.clicked.connect(self.player.play)
        self.pauseButton.clicked.connect(self.player.pause)
        self.stopButton.clicked.connect(self.player.stop)
        self.volumeSlider.valueChanged.connect(lambda value: self.audioOutput.setVolume(value / 100))
        self.videoSlider = QSlider(Qt.Orientation.Horizontal)
        self.videoSlider.sliderMoved.connect(self.setPosition)
        self.player.positionChanged.connect(self.updatePosition)
        self.player.durationChanged.connect(self.updateDuration)
        # Layout for the video player controls
        controlsLayout = QHBoxLayout()
        controlsLayout.addWidget(self.playButton)
        controlsLayout.addWidget(self.pauseButton)
        controlsLayout.addWidget(self.stopButton)
        controlsLayout.addWidget(QLabel("Volume:"))
        controlsLayout.addWidget(self.volumeSlider)

        # Main layout for the video subwindow
        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.videoWidget)
        mainLayout.addWidget(self.videoSlider)

        mainLayout.addLayout(controlsLayout)

        self.videoSubWindow.widget().setLayout(mainLayout)
    def tileMdiSubWindows(self):
        """Resize and evenly distribute all subwindows in the MDI area."""
        # Optionally, set a uniform size for all subwindows
        subwindows = self.mdiArea.subWindowList()
        if not subwindows:
            return

        # Calculate the optimal size for subwindows based on the number of windows
        areaSize = self.mdiArea.size()
        numWindows = len(subwindows)
        width = int(areaSize.width() / numWindows ** 0.5)
        height = int(areaSize.height() / numWindows ** 0.5)

        # Resize each subwindow (this step is optional)
        for window in subwindows:
            window.resize(width, height)

        # Tile the subwindows
        self.mdiArea.tileSubWindows()
    def setPosition(self, position):
        self.player.setPosition(position)

    def updatePosition(self, position):
        self.videoSlider.setValue(position)

    def updateDuration(self, duration):
        self.videoSlider.setMaximum(duration)

    @pyqtSlot(str)
    def playVideo(self, path):
        # Play video in the QMediaPlayer
        self.player.setSource(QUrl.fromLocalFile(path))
        self.player.play()
        self.videoSubWindow.show()  # Ensure the video subwindow is visible

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
        self.thread.finished.connect(self.playVideo)
        self.thread.start()
    def stopBackendProcess(self):
        try:
            from deforum.shared_storage import models
            models["deforum_pipe"].gen.max_frames = len(models["deforum_pipe"].images)
        except:
            pass


    def startBatchProcess(self):
        if not self.current_job and self.job_queue:
            self.runNextJob()

    def runNextJob(self):
        if self.job_queue:
            self.current_job = self.job_queue.pop(0)
            self.current_job.markRunning()  # Mark the job as running
            self.thread = BackendThread(self.current_job.job_data)
            self.thread.finished.connect(self.onJobFinished)
            self.thread.start()

    @pyqtSlot(str)
    def onJobFinished(self, result):
        print(f"Job completed with result: {result}")
        self.current_job.markComplete()  # Mark the job as complete
        self.current_job = None  # Reset current job
        self.runNextJob()  # Run next job in the queue

    def stopBatchProcess(self):
        if self.current_job:
            # Logic to stop the current thread if it's running
            self.thread.terminate()
            self.current_job.markComplete()
            self.current_job = None
        while self.job_queue:
            job = self.job_queue.pop(0)
            job.markComplete()

    @pyqtSlot(dict)
    def updateImage(self, data):
        # Update the image on the label
        if 'image' in data:
            img = copy.deepcopy(data['image'])
            qpixmap = npArrayToQPixmap(np.array(img).astype(np.uint8))  # Convert to QPixmap
            # Get the size of the container QMdiSubWindow
            container_size = self.previewSubWindow.size()

            # Scale the QPixmap to fit within the container while maintaining the aspect ratio
            scaled_pixmap = qpixmap.scaled(container_size, Qt.AspectRatioMode.KeepAspectRatio,
                                           Qt.TransformationMode.SmoothTransformation)

            # Set the scaled pixmap to the label
            self.previewLabel.setPixmap(scaled_pixmap)
            self.previewLabel.setScaledContents(True)
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
