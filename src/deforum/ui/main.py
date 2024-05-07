import sys

from qtpy.QtWidgets import QApplication

from deforum.ui.qt_modules.main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec())
