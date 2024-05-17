import os
import platform
import sys

from PyQt6.QtGui import QIcon
from qtpy.QtWidgets import QApplication

from deforum.ui.qt_modules.main_window import MainWindow
from deforum.utils.constants import config

def is_dark_mode():
    """Detect if the system is in dark mode."""
    try:
        if platform.system() == 'Windows':
            import winreg
            registry = winreg.ConnectRegistry(None, winreg.HKEY_CURRENT_USER)
            key = winreg.OpenKey(registry, r'Software\Microsoft\Windows\CurrentVersion\Themes\Personalize')
            value, _ = winreg.QueryValueEx(key, 'AppsUseLightTheme')
            return value == 0
        elif platform.system() == 'Darwin':  # macOS
            import subprocess
            result = subprocess.run(['defaults', 'read', '-g', 'AppleInterfaceStyle'], capture_output=True, text=True)
            return 'Dark' in result.stdout
        else:  # Assume Linux or other Unix-like system
            if 'XDG_CURRENT_DESKTOP' in os.environ:
                desktop = os.environ['XDG_CURRENT_DESKTOP'].lower()
                if 'gnome' in desktop or 'kde' in desktop:
                    import subprocess
                    result = subprocess.run(['gsettings', 'get', 'org.gnome.desktop.interface', 'gtk-theme'], capture_output=True, text=True)
                    return 'dark' in result.stdout.lower()
    except Exception:
        return False
    return False

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    # Platform-specific adjustments
    if platform.system() == "Windows":
        import ctypes

        myappid = 'deforum.app'  # Arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    # Set the window and taskbar icon
    icon_path = os.path.join(config.src_path, 'deforum', 'ui', 'deforum.png')  # Adjust the path if the icon is located elsewhere
    app.setWindowIcon(QIcon(icon_path))  # Set the icon for the taskbar on some platforms
    mainWindow.setWindowIcon(QIcon(icon_path))  # Set the icon for the window
    # Load the appropriate QSS file based on the system theme
    if is_dark_mode():
        qss_path = os.path.join(config.src_path, 'deforum', 'ui', 'qss', 'dark.qss')
    else:
        qss_path = os.path.join(config.src_path, 'deforum', 'ui', 'qss', 'light.qss')

    with open(qss_path, 'r') as file:
        qss = file.read()
        app.setStyleSheet(qss)

    mainWindow.show()
    sys.exit(app.exec())
