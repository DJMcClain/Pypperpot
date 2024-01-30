from PyQt5.QtWidgets import *
import sys
import Mainapp

app = QApplication(sys.argv)
app.setStyle('Fusion')
w = Mainapp.MainWindow()
w.show()
app.exec()













