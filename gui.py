import sys
from PyQt5.QtWidgets import QApplication, QWidget

app = QApplication(sys.argv)
w = QWidget()
w.resize(500, 500)
w.move(500, 500)
w.show()
w.setWindowTitle("deep learning")
sys.exit(app.exec_())
