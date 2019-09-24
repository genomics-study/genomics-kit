import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot

class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'Genomics Studies - Dashboard'
        self.left = 10
        self.top = 10
        self.width = 1040
        self.height = 800
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
    
        # Create text label
        self.l1 = QLabel(self)
        self.l1.move(200, 705)
        self.l1.resize(300, 20)
        self.l1.setText("")
        
        self.l2 = QLabel(self)
        self.l2.move(430, 20)
        self.l2.resize(300,20)
        self.l2.setText("Insert model description")

        # Create textbox
        self.textbox = QPlainTextEdit(self)
        self.textbox.move(20, 60)
        self.textbox.resize(1000,600)
        
        # Create a button in the window
        self.button = QPushButton('Evaluate', self)
        self.button.move(850,700)

	# Create a button in the window
        self.button2 = QPushButton('Load Dataset', self)
        self.button2.move(80,700)
        
        # connect button to function on_click
        self.button.clicked.connect(self.on_click)
        self.button2.clicked.connect(self.openFileNameDialog)

        self.show()

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;CSV Files (*.csv)", options=options)
        if fileName:
            self.l1.setText(fileName)

    @pyqtSlot()
    def on_click(self):
        textboxValue = self.textbox.toPlainText()
        # TODO run our model here and print result
        QMessageBox.question(self, 'Genomics Studies - Summary', str, QMessageBox.Ok, QMessageBox.Ok)

str = """
array([[ 0.        , -0.        ,  0.41497871,  0.        ],
       [-0.13847884,  0.00695683,  0.2958312 ,  0.50294318],
       [ 0.09266584,  0.00651644,  0.46567182,  0.4351459 ],
       [ 3.        ,  2.        ,  1.        ,  1.        ],
       [ 2.        ,  3.        ,  1.        ,  1.        ]])
"""

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
