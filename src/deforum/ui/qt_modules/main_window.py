import copy
import datetime
import os
import sys
import json

import numpy as np

from PyQt6.QtGui import QImage, QAction
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import QApplication, QTabWidget, QWidget, QVBoxLayout, QDockWidget, QSlider, QLabel, QMdiArea, \
    QPushButton, QComboBox, QFileDialog, QSpinBox, QLineEdit, QCheckBox, QTextEdit, QHBoxLayout, QListWidget, \
    QDoubleSpinBox, QListWidgetItem, QMessageBox, QScrollArea, QTabBar
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QUrl, QSize

from deforum import logger
from deforum.ui.qt_helpers.qt_image import npArrayToQPixmap
from deforum.ui.qt_modules.backend_thread import BackendThread
from deforum.ui.qt_modules.core import DeforumCore
from deforum.ui.qt_modules.custom_ui import ResizableImageLabel, JobDetailPopup, JobQueueItem, AspectRatioMdiSubWindow
from deforum.ui.qt_modules.ref import TimeLineQDockWidget
from deforum.ui.qt_modules.timeline import TimelineWidget


# class DetachableTabWidget(QTabWidget):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setMovable(True)
#         self.setTabBarAutoHide(False)
#         self.tabBar().setMouseTracking(True)
#         self.main_window = parent  # Assume parent is the main window
#
#     def addTabWithDetachButton(self, widget, title):
#         super().addTab(widget, title)
#         tab_index = self.indexOf(widget)
#         detach_button = QPushButton("Detach")
#         detach_button.setProperty('widget', widget)  # Store the widget directly
#         detach_button.setProperty('title', title)  # Store the widget directly
#         detach_button.clicked.connect(self.detach_as_dock)
#         self.tabBar().setTabButton(tab_index, QTabBar.ButtonPosition.RightSide, detach_button)
#
#     def detach_as_dock(self):
#         button = self.sender()
#         if button:
#             widget = button.property('widget')
#             if widget is None:
#                 return
#
#             index = self.indexOf(widget)  # Determine index at the time of click
#             if index == -1:
#                 return  # This should not happen, but just in case
#
#             scroll_area = QScrollArea()
#             scroll_area.setWidget(widget)
#             scroll_area.setWidgetResizable(True)
#
#             title = button.property('title')
#             self.dock = AutoReattachDockWidget(title, self.main_window, widget, self)
#             self.dock.setObjectName(title)
#             self.dock.setWidget(scroll_area)
#             self.dock.setFloating(True)
#             self.dock.show()
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
class MainWindow(DeforumCore):
    def __init__(self):
        super().__init__()
        self.player = QMediaPlayer()  # Create a media player object
        self.setWindowTitle("Deforum Engine")
        self.setupDynamicUI()
        self.currentTrack = None  # Store the currently selected track
        self.presets_folder = os.path.join(os.path.expanduser('~'), 'deforum', 'presets')
        self.projects_folder = os.path.join(os.path.expanduser('~'), 'deforum', 'projects')
        os.makedirs(self.presets_folder, exist_ok=True)
        os.makedirs(self.projects_folder, exist_ok=True)
        self.loadPresetsDropdown()
        self.presetsDropdown.activated.connect(self.loadPresetsDropdown)
        self.setupVideoPlayer()
        self.tileMdiSubWindows()
        self.timelineDock.hide()
        self.setupBatchControls()
        self.job_queue = []  # List to keep jobs to be processed
        self.current_job = None  # To keep track of the currently running job
        self.setupMenu()
        self.newProject()
        self.state_file = os.path.join(os.path.expanduser('~'), 'deforum', '.lastview')
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        self.loadWindowState()
    def closeEvent(self, event):
        self.saveWindowState()
        super().closeEvent(event)

    def saveWindowState(self):
        with open(self.state_file, 'w') as f:
            geometry_base64 = self.saveGeometry().toBase64().data().decode('utf-8')
            state_base64 = self.saveState().toBase64().data().decode('utf-8')
            f.write(geometry_base64 + '\n')
            f.write(state_base64 + '\n')

            for dock in self.findChildren(AutoReattachDockWidget):
                dock_state = {
                    'name': dock.objectName(),
                    'should_detach':dock.should_detach,
                    'floating': dock.isFloating(),
                    'geometry': dock.saveGeometry().toBase64().data().decode('utf-8'),
                    'visible': dock.isVisible(),
                    'is_detached': dock.isFloating(),  # or any other logic you determine
                    'dock_area': str(self.dockWidgetArea(dock)).split('.')[1]
                }
                json.dump(dock_state, f)
                f.write('\n')

    def loadWindowState(self):
        try:
            with open(self.state_file, 'r') as f:
                self.restoreGeometry(bytes(f.readline().strip(), 'utf-8'))
                self.restoreState(bytes(f.readline().strip(), 'utf-8'))

                while True:
                    line = f.readline().strip()
                    if not line:
                        break
                    dock_state = json.loads(line)

                    for button in self.tabWidget.buttons:
                        if button.property('title') == dock_state['name']:
                            button.click()

                    dock = self.findChild(AutoReattachDockWidget, dock_state['name'])

                    area_map = {
                        "TopDockWidgetArea": Qt.DockWidgetArea.TopDockWidgetArea,
                        "BottomDockWidgetArea": Qt.DockWidgetArea.BottomDockWidgetArea,
                        "LeftDockWidgetArea": Qt.DockWidgetArea.LeftDockWidgetArea,
                        "RightDockWidgetArea": Qt.DockWidgetArea.RightDockWidgetArea,
                        "NoDockWidgetArea": Qt.DockWidgetArea.NoDockWidgetArea
                    }
                    if dock:
                        dock.restoreGeometry(bytes(dock_state['geometry'], 'utf-8'))
                        self.addDockWidget(area_map[dock_state['dock_area']], dock)
                        dock.setFloating(False)
        except FileNotFoundError:
            print("No saved state to load.")
        except Exception as e:
            print(f"Failed to load state: {e}")

    def setupMenu(self):
        menuBar = self.menuBar()

        fileMenu = menuBar.addMenu('&File')

        newProjectAction = QAction('&New Project', self)
        newProjectAction.triggered.connect(self.newProject)
        fileMenu.addAction(newProjectAction)

        loadProjectAction = QAction('&Load Project', self)
        loadProjectAction.triggered.connect(self.loadProject)
        fileMenu.addAction(loadProjectAction)

        saveProjectAction = QAction('&Save Project', self)
        saveProjectAction.triggered.connect(self.saveProject)
        fileMenu.addAction(saveProjectAction)

        convertProjectAction = QAction('&Convert Project', self)
        convertProjectAction.triggered.connect(self.convertProjectToSingleSettingsFile)
        fileMenu.addAction(convertProjectAction)

    def newProject(self):
        # Reset the current project
        # self.params = {}  # Reset or set to default values
        # self.job_queue = []  # Clear any existing jobs
        # self.current_job = None
        # self.jobQueueList.clear()
        self.project = {
            'params': {},  # Global parameters if any
            'frames': {}  # Dictionary to store parameters per frame
        }
        # Reset other necessary parts of the UI and data

    def loadProject(self):
        # Load project settings and parameters from a file
        fname = QFileDialog.getOpenFileName(self, 'Open Project File', self.projects_folder, 'Deforum Project Files (*.dproj)')
        if fname[0]:
            with open(fname[0], 'r') as file:
                self.project = json.load(file)  # Load the entire project dictionary
                if 'frames' in self.project and self.project['frames']:
                    # Assuming frames are stored with an index key that starts at 0
                    self.params = self.project['frames'].get('1', self.params)
                # else:
                #     self.params = self.project.get('params', {})
                    self.updateUIFromParams()  # Update UI elements with loaded params

    def saveProject(self):
        # Save current settings and entire project data to a file
        fname = QFileDialog.getSaveFileName(self, 'Save Project File', self.projects_folder, 'Deforum Project Files (*.dproj)')
        if fname[0]:
            if not fname[0].endswith('.dproj'):
                fname = fname[0] + '.dproj'  # Ensure the file has the correct extension
            with open(fname, 'w') as file:
                json.dump(self.project, file, indent=4)  # Save the entire project dictionary

    def convertProjectToSingleSettingsFile(self):
        # Step 1: Load a .dproj project file
        fname = QFileDialog.getOpenFileName(self, 'Open Project File', '', 'Deforum Project Files (*.dproj)')
        if not fname[0]:
            return

        with open(fname[0], 'r') as file:
            self.project = json.load(file)

        if 'frames' not in self.project or not self.project['frames']:
            QMessageBox.warning(self, "Error", "No frame data found in the project.")
            return

        # Step 2: Use Frame 1 as a base configuration
        base_params = self.project['frames']['1'].copy()
        max_frame_index = max(map(int, self.project['frames'].keys()))

        # Initialize parameter consolidation
        consolidated_params = {}
        for param in base_params:
            if isinstance(base_params[param], str) and ':' in base_params[param]:
                consolidated_params[param] = {}

        # Step 3: Collect and consolidate parameter values from all frames
        for frame_index in range(1, max_frame_index + 1):
            frame_key = str(frame_index)
            if frame_key in self.project['frames']:
                for param in consolidated_params:
                    value = self.project['frames'][frame_key].get(param, "")
                    if value and ':' in value:  # Check if value fits the pattern requiring consolidation
                        # Extract value and append to the schedule
                        try:
                            schedule_part = value.split(':')[-1].strip()
                            consolidated_params[param][frame_index - 1] = schedule_part
                        except IndexError:
                            continue

        # Integrate consolidated parameters into base_params
        for param, schedule in consolidated_params.items():
            if schedule:
                schedule_str = ", ".join(f"{k}: {v}" for k, v in schedule.items())
                base_params[param] = schedule_str

        # Step 4: Save the consolidated settings to a .txt file
        fname = QFileDialog.getSaveFileName(self, 'Save Consolidated Settings File', '', 'Text Files (*.txt)')
        if fname[0]:
            with open(fname[0], 'w') as file:
                json.dump(base_params, file, indent=4)

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
        loadFilesButton = QPushButton("Load Batch Files")
        startButton = QPushButton("Start Batch")
        cancelButton = QPushButton("Cancel Batch")

        # Connect buttons to their respective slots (functions)
        startButton.clicked.connect(self.startBatchProcess)
        cancelButton.clicked.connect(self.stopBatchProcess)
        loadFilesButton.clicked.connect(self.loadBatchFiles)
        layout.addWidget(loadFilesButton)
        layout.addWidget(startButton)
        layout.addWidget(cancelButton)
        layout.addWidget(self.jobQueueList)

        # Add the batch control tab to the main tab widget
        self.tabWidget.addTab(self.batchControlTab, "Batch Control")

    def loadBatchFiles(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setNameFilter("Text Files (*.txt)")
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            for file_path in selected_files:
                self.createJobFromConfig(file_path)
    def createJobFromConfig(self, file_path):
        try:
            with open(file_path, 'r') as file:
                job_data = json.load(file)
                batch_name = os.path.basename(file_path)
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                job_name = f"{batch_name} {timestamp}"
                self.addCurrentParamsAsJob(job_name, job_data)
        except json.JSONDecodeError as e:
            print(f"Error loading {file_path}: {e}")
            QMessageBox.warning(self, "Loading Error", f"Failed to load {file_path}: {e}")
    def addCurrentParamsAsJob(self, job_name=None, job_params=None):
        if not job_name:
            job_name = f"{self.params.get('batch_name')}_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            job_params = self.params
        # Adding current params as a new job
        # job_description = json.dumps(self.params, indent=4)

        item = QListWidgetItem(self.jobQueueList)
        widget = JobQueueItem(job_name, job_params)
        item.setSizeHint(widget.sizeHint())
        self.jobQueueList.addItem(item)
        self.jobQueueList.setItemWidget(item, widget)
        self.job_queue.append(widget)  # Append job widget to the queue
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
        self.toolbar.setObjectName('main_toolbar')
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
        self.tabWidget = DetachableTabWidget(self)
        self.tabWidget.setMinimumSize(QSize(0,0))
        self.controlsDock.setWidget(self.tabWidget)

        # Load UI configuration from JSON
        curr_folder = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(curr_folder, 'ui.json')
        with open(json_path, 'r') as file:
            config = json.load(file)

        for category, settings in config.items():
            tab = QWidget()  # This is the actual tab that will hold the layout
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
                    self.createTextInput(params['label'], layout, params['default'], setting)
                elif params['widget_type'] == 'text box':
                    self.createTextBox(params['label'], layout, params['default'], setting)
                elif params['widget_type'] == 'slider':
                    self.createSlider(params['label'], layout, params['min'], params['max'], params['default'], setting)
                elif params['widget_type'] == 'checkbox':
                    self.createCheckBox(params['label'], layout, bool(params['default']), setting)
                elif params['widget_type'] == 'file_input':
                    # Add file input handler if needed
                    pass
            scroll = QScrollArea()  # Create a scroll area
            scroll.setWidget(tab)
            scroll.setWidgetResizable(True)  # Make the scroll area resizable

            self.tabWidget.addTabWithDetachButton(scroll, category)  # Add the scroll area to the tab widget
            # self.tabWidget.addTab(tab, category)
        # self.createPushButton('Render', layout, self.startBackendProcess)

        # Create preview area
        self.setupPreviewArea()

        # self.timeline = TimelineWidget()
        # self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.timeline)
        # Create the timeline dock widget
        self.timelineDock = TimeLineQDockWidget(self)
        self.timelineDock.setObjectName('timeline_dock')
        # self.timelineDock = QDockWidget("Timeline", self)
        self.timelineDock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)

        # Create and set the timeline widget inside the dock
        # self.timelineWidget = TimelineWidget()
        # self.timelineDock.setWidget(self.timelineWidget)

        # Add the dock widget to the main window
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.timelineDock)

    def setupPreviewArea(self):
        # Setting up an MDI area for previews
        self.mdiArea = QMdiArea()
        self.setCentralWidget(self.mdiArea)

        self.previewLabel = ResizableImageLabel()
        self.previewSubWindow = AspectRatioMdiSubWindow(self.previewLabel)
        self.mdiArea.addSubWindow(self.previewSubWindow)
        self.previewSubWindow.setWidget(self.previewLabel)
        self.previewSubWindow.show()


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
        self.videoSlider.valueChanged.connect(self.setResumeFrom)

    def setResumeFrom(self, position):
        # Assuming the frame rate is stored in self.frameRate
        frame_rate = self.params.get('fps', 24)  # you'll need to set this appropriately
        frame_number = int((position / 1000) * frame_rate)
        self.params["resume_from"] = frame_number + 1
        # Update the widget directly if necessary
        if 'resume_from' in self.widgets:
            self.widgets['resume_from'].setValue(frame_number)

    # def setResumeFrom(self, to):
    #     # Assuming position is in milliseconds and you want to convert it to seconds
    #     seconds = to // 1000
    #     self.params["resume_from"] = seconds
    #     # Update the widget directly if necessary
    #     if 'resume_from' in self.widgets:
    #         self.widgets['resume_from'].setValue(seconds)
    #
    #     # self.params["resume_from"] = to
    #     # self.widgets['resume_from'].setValue(to)

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
        # Optionally, you might want to adjust the ticks interval based on the video duration
        self.videoSlider.setTickInterval(duration // 100)  # For example, 100 ticks across the slider

    @pyqtSlot(dict)
    def playVideo(self, data):
        # Stop the player and reset its state before loading a new video
        self.player.stop()
        self.player.setSource(QUrl())  # Reset the source to clear buffers

        if 'video_path' in data:
            try:
                # Set new video source and attempt to play
                self.player.setSource(QUrl.fromLocalFile(data['video_path']))
                self.player.play()
            except Exception as e:
                print(f"Error playing video: {e}")
                self.player.stop()  # Ensure player is stopped on error

        if 'timestring' in data:
            self.params['resume_timestring'] = data['timestring']
            self.params['resume_path'] = data['resume_path']
            self.params['resume_from'] = data['resume_from']
            self.updateUIFromParams()

    # def addDock(self, title, area):
    #     dock = QDockWidget(title, self)
    #     dock.setMinimumSize(QSize(0, 0))  # Allow resizing to very small sizes
    #     self.addDockWidget(area, dock)
    #     return dock
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

    def startBackendProcess(self):
        #params = {key: widget.value() for key, widget in self.params.items() if hasattr(widget, 'value')}
        # if not self.thread:

        if self.params['resume_from_timestring']:
            self.params['max_frames'] += 1
            self.updateUIFromParams()

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
            try:
                self.current_job.markRunning()  # Mark the job as running
                # self.params.update(self.current_job.job_data)
                # self.updateUIFromParams()
                self.thread = BackendThread(self.current_job.job_data)
                self.thread.finished.connect(self.onJobFinished)
                self.thread.imageGenerated.connect(self.updateImage)
                # self.thread.finished.connect(self.playVideo)
                self.thread.start()
            except:
                self.current_job = None
                self.startBatchProcess()

    @pyqtSlot(dict)
    def onJobFinished(self, result):
        #print(f"Job completed with result: {result}")
        if self.current_job:
            self.current_job.markComplete()  # Mark the job as complete
            self.current_job = None  # Reset current job
            self.runNextJob()  # Run next job in the queue
        else:
            self.jobQueueList.clear()

    def stopBatchProcess(self):
        if self.current_job:
            # Logic to stop the current thread if it's running
            try:
                from deforum.shared_storage import models
                if 'deforum_pipe' in models:
                    models["deforum_pipe"].gen.max_frames = len(models["deforum_pipe"].images)
            except Exception as e:
                logger.info(repr(e))
        else:
            self.jobQueueList.clear()
        self.current_job = None


    @pyqtSlot(dict)
    def updateImage(self, data):
        # Update the image on the label
        if 'image' in data:
            img = copy.deepcopy(data['image'])
            qpixmap = npArrayToQPixmap(np.array(img).astype(np.uint8))  # Convert to QPixmap
            self.previewLabel.setPixmap(qpixmap)
            self.previewLabel.setScaledContents(True)
            if self.project is not None:
                # Save parameters for this frame

                # Save parameters for this frame
                self.project['frames'][data['frame_idx']] = copy.deepcopy(self.params)

                # If there are subsequent frame parameters saved, load them
                if data['frame_idx'] + 1 in self.project['frames']:
                    self.params = copy.deepcopy(self.project['frames'][data['frame_idx'] + 1])
                # else:
                #     # Reset to global defaults if no specific frame data exists
                #     self.params = copy.deepcopy(self.project.get('params', {}))

                    self.updateUIFromParams()  # Reflect parameter updates in UI

                # self.project['frames'][data['frame_idx']] = copy.deepcopy(self.params)
                #
                # # Prepare params for the next frame
                # if data['frame_idx'] + 1 in self.project['frames']:
                #     self.params = copy.deepcopy(self.project['frames'][data['frame_idx'] + 1])
                #     self.updateUIFromParams()
            # self.timelineWidget.add_image_to_track(qpixmap)
    def updateParam(self, key, value):
        super().updateParam(key, value)
        try:
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
        except:
            pass
    def updateUIFromParams(self):
        for k, widget in self.widgets.items():
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