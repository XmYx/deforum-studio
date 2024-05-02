import sys
from PyQt6.QtCore import QCoreApplication
from argparse import ArgumentParser

from deforum.ui.qt_modules.backend_thread import BackendThread


def main(settings_file):
    # Initialize the Qt application
    app = QCoreApplication(sys.argv)

    # Parameters dictionary to pass to the BackendThread
    params = {
        'settings_file': settings_file
    }

    # Create an instance of BackendThread
    backend_thread = BackendThread(params)

    # Connect signals to handlers
    backend_thread.imageGenerated.connect(image_generated_handler)
    backend_thread.finished.connect(finished_handler)

    # Start the thread
    backend_thread.start()

    # Execute the Qt application
    sys.exit(app.exec())


def image_generated_handler(image_data):
    print("Image data received:", image_data)


def finished_handler(status):
    print("Process finished with status:", status)
    QCoreApplication.quit()


if __name__ == '__main__':
    parser = ArgumentParser(description="Run the BackendThread with a settings file.")
    parser.add_argument('settings_file', type=str, help='Path to the settings file.')

    args = parser.parse_args()

    main(args.settings_file)
