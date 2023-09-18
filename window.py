import sys

# pyqt5
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QApplication
from PyQt5.QtGui import QFont
from PyQt5.uic import loadUi

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

    def classify_leukocyte(self):      
        # Obter o caminho do arquivo da imagem
        file_name = self.input_text.text()

        if file_name:
            class_label = self.predictor.detect_and_contour_cell(file_name)
            self.resultado_text.setPlainText(class_label)    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
