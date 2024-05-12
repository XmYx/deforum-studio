import sys
import os
from threading import Thread
from qtpy.QtCore import QCoreApplication, Signal, QObject
from argparse import ArgumentParser

from deforum.ui.qt_modules.backend_thread import BackendThread


class ApplicationManager(QObject):
    start_new_thread = Signal(str)  # Signal to start a new BackendThread with a settings file

    def __init__(self, folder_path):
        super().__init__()
        self.app = QCoreApplication(sys.argv)
        self.folder_path = folder_path
        self.start_new_thread.connect(self.run_backend_thread)

    def run_backend_thread(self, settings_file):
        if not os.path.exists(settings_file):
            print(f"File {settings_file} does not exist. Please enter a valid file path.")
            return

        params = {'settings_file': settings_file}
        self.backend_thread = BackendThread(params)
        self.backend_thread.imageGenerated.connect(self.image_generated_handler)
        self.backend_thread.finished.connect(self.finished_handler)
        self.backend_thread.start()

    def image_generated_handler(self, image_data):
        # Handle generated image data here
        print("Image data received:", image_data)

    def finished_handler(self, status):
        print("Process finished with status:", status)
        self.process_next_file()

    def process_files(self):
        self.file_list = [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if f.endswith('.txt')]
        self.current_file_index = 0
        if self.file_list:
            self.process_next_file()

    def process_next_file(self):
        if self.current_file_index < len(self.file_list):
            self.start_new_thread.emit(self.file_list[self.current_file_index])
            self.current_file_index += 1
        else:
            print("All files processed.")
            QCoreApplication.quit()

    def exec(self):
        self.process_files()
        return self.app.exec()

if __name__ == '__main__':
    parser = ArgumentParser(description="Run the BackendThread with a directory of settings files.")
    parser.add_argument('folder_path', type=str, help='Path to the folder containing settings files.')
    args = parser.parse_args()

    manager = ApplicationManager(args.folder_path)
    sys.exit(manager.exec())
