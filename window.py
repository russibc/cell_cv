import sys

# pyqt5
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QApplication
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.uic import loadUi
from PyQt5 import QtCore

# proprias
from image_predictor import ImagePredictor

class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        loadUi("window.ui", self)

        font = QFont()
        font.setBold(True)
        self.input_label.setFont(font)
        self.resultado_label.setFont(font)

        self.button.clicked.connect(self.open_file_dialog)
        self.generate_button.clicked.connect(self.classify_leukocyte)

        self.predictor = ImagePredictor()

    def open_file_dialog(self):
        options = QFileDialog.Options()
        options = QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Abrir arquivo",
            "",
            "Imagens (*.jpg *.jpeg *.png)",
            options=options,
        )
        if file_name:
            self.input_text.setText(file_name)
            self.display_image(file_name)

    def classify_leukocyte(self):      
        # Obter o caminho do arquivo da imagem
        file_name = self.input_text.text()

        if file_name:
            class_label, output_file_with_contour = self.predictor.detect_and_contour_cell(file_name)
            self.display_image(output_file_with_contour)
            self.resultado_text.setPlainText(class_label)

    def display_image(self, output_file):     
        pixmap = QPixmap(output_file)
        pixmap = pixmap.scaled(100, 100)
        self.thumbnail_label.setPixmap(pixmap)
        self.thumbnail_label.setScaledContents(True)

        # Centralizar imagem dentro do QLabel
        self.thumbnail_label.setAlignment(QtCore.Qt.AlignCenter)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
