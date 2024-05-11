# import sys
# from qtpy.QtCore import QCoreApplication
# from argparse import ArgumentParser
#
# from deforum.ui.qt_modules.backend_thread import BackendThread
#
#
# def main(settings_file):
#     # Initialize the Qt application
#     app = QCoreApplication(sys.argv)
#
#     # Parameters dictionary to pass to the BackendThread
#     params = {
#         'settings_file': settings_file
#     }
#
#     # Create an instance of BackendThread
#     backend_thread = BackendThread(params)
#
#     # Connect signals to handlers
#     backend_thread.imageGenerated.connect(image_generated_handler)
#     backend_thread.finished.connect(finished_handler)
#
#     # Start the thread
#     backend_thread.start()
#
#     # Execute the Qt application
#     sys.exit(app.exec())
#
#
# def image_generated_handler(image_data):
#     print("Image data received:", image_data)
#
#
# def finished_handler(status):
#     print("Process finished with status:", status)
#     QCoreApplication.quit()
#
#
# if __name__ == '__main__':
#     parser = ArgumentParser(description="Run the BackendThread with a settings file.")
#     parser.add_argument('settings_file', type=str, help='Path to the settings file.')
#
#     args = parser.parse_args()
#
#     main(args.settings_file)
import sys
import os
from threading import Thread
from qtpy.QtCore import QCoreApplication, Signal, QObject
from argparse import ArgumentParser

from deforum.ui.qt_modules.backend_thread import BackendThread


class ApplicationManager(QObject):
    start_new_thread = Signal(str)  # Signal to start a new BackendThread with a settings file

    def __init__(self):
        super().__init__()
        self.app = QCoreApplication(sys.argv)
        self.start_new_thread.connect(self.run_backend_thread)

    def run_backend_thread(self, settings_file):
        if not os.path.exists(settings_file):
            # print(f"File {settings_file} does not exist. Please enter a valid file path.")
            return

        params = {
            'settings_file': settings_file
        }
        self.backend_thread = BackendThread(params)
        self.backend_thread.imageGenerated.connect(self.image_generated_handler)
        self.backend_thread.finished.connect(self.finished_handler)
        self.backend_thread.start()

    def image_generated_handler(self, image_data):
        pass
        # print("Image data received:", image_data)

    def finished_handler(self, status):
        print("Process finished with status:", status)
        self.request_user_input()

    def request_user_input(self):
        # Start a thread to request user input to keep the Qt event loop unblocked
        input_thread = Thread(target=self.user_input_handler)
        input_thread.start()

    def user_input_handler(self):
        print("Enter the path to a new settings file to continue rendering, or just press Enter to exit:")
        path = input().strip()
        if path:
            self.start_new_thread.emit(path)
        else:
            QCoreApplication.quit()

    def exec(self):
        return self.app.exec()


if __name__ == '__main__':
    parser = ArgumentParser(description="Run the BackendThread with a settings file.")
    parser.add_argument('settings_file', type=str, help='Path to the settings file.')

    args = parser.parse_args()

    manager = ApplicationManager()
    manager.start_new_thread.emit(args.settings_file)
    sys.exit(manager.exec())
