import sys
import os

from PyQt6.QtGui import QAction, QWindow, QPalette
from PyQt6.QtWidgets import QApplication, QMainWindow, QTreeWidget, QTreeWidgetItem, QTextBrowser, QVBoxLayout, \
    QSplitter, QDialog, QLineEdit, QLabel, QTreeWidgetItemIterator
from PyQt6.QtCore import Qt

from deforum.utils.constants import config


class HelpDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Help")
        self.setMinimumSize(800, 600)


        # # Create the search bar
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search...")
        self.search_bar.textChanged.connect(self.search_help)
        #
        # Create the tree widget for topics
        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("Topics")
        self.tree.itemClicked.connect(self.display_help_content)
        # Add Home link
        self.add_home_link()
        # # Populate the tree with topics and subtopics
        self.add_topics()
        #
        # Create the text browser for displaying help content
        self.text_browser = QTextBrowser()

        # Load the start page
        self.load_start_page()
        #
        # # Layout for the help dialog
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.tree)
        splitter.addWidget(self.text_browser)
        splitter.setSizes([int(self.width() * 0.2), int(self.width() * 0.8)])  # Set the sizes as [tree width, text browser width]


        layout = QVBoxLayout()
        layout.addWidget(self.search_bar)
        layout.addWidget(splitter)
        self.setLayout(layout)
    def add_home_link(self):
        home_item = QTreeWidgetItem(self.tree, ["Home"])
        home_item.setData(0, Qt.ItemDataRole.UserRole, "index.html")

    def add_topics(self):
        help_dir = os.path.join(config.src_path, 'deforum', 'ui', 'help_content')
        for root, dirs, files in os.walk(help_dir):
            parent_item = self.tree
            # if root != help_dir:
            #     parent_name = os.path.basename(root)
            #     parent_item = self.find_or_create_item(parent_item, parent_name)

            for file in files:
                if file.endswith(".html") and file != "index.html":
                    file_name = os.path.splitext(file)[0]
                    self.find_or_create_item(parent_item, file_name, os.path.join(root, file))

    def find_or_create_item(self, parent_item, name, path=None):
        # for i in range(parent_item.childCount()):
        #     if parent_item.child(i).text(0) == name:
        #         return parent_item.child(i)

        new_item = QTreeWidgetItem(parent_item, [name])
        if path:
            new_item.setData(0, Qt.ItemDataRole.UserRole, path)
        return new_item

    def adjust_html_for_theme(self, html_content):
        # Check if dark mode is active
        if self.palette().color(QPalette.ColorRole.Window).lightness() < 128:
            # Add CSS to invert colors for dark mode
            return f"<style>body {{ filter: invert(100%); }}</style>{html_content}"
        return html_content

    def display_help_content(self, item, column):
        file_path = item.data(0, Qt.ItemDataRole.UserRole)
        if file_path and os.path.exists(file_path):
            with open(file_path, "r") as file:
                content = file.read()
                adjusted_content = self.adjust_html_for_theme(content)
                self.text_browser.setHtml(adjusted_content)
        else:
            self.text_browser.setHtml(self.adjust_html_for_theme("<h1>Help</h1><p>Select a topic to view help content.</p>"))


    def load_start_page(self):
        index_path = os.path.join(config.src_path, 'deforum', 'ui', 'help_content', 'index.html')
        if os.path.exists(index_path):
            with open(index_path, "r") as file:
                content = file.read()
                self.text_browser.setHtml(content)
        else:
            self.text_browser.setHtml("<h1>Help</h1><p>Welcome to the help system.</p>")

    def search_help(self, text):
        iterator = QTreeWidgetItemIterator(self.tree)
        while iterator.value():
            item = iterator.value()
            item.setHidden(True)
            if text.lower() in item.text(0).lower():
                item.setHidden(False)
                parent = item.parent()
                while parent:
                    parent.setHidden(False)
                    parent = parent.parent()
            iterator += 1


class AboutDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("About")
        layout = QVBoxLayout()
        about_label = QLabel("This is an example PyQt6 application.\n\n"
                             "Version 1.0\n"
                             "Developed by Your Name.")
        layout.addWidget(about_label)
        self.setLayout(layout)
