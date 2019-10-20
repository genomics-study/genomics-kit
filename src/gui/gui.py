import re
import sys

import numpy as np
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import *
from pandas.io import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from src.model.model import Model, LogisticRegressionCV, RFECV
from src.preprocessing.data_sim import data_sim
from src.validation.CV import Validator


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
        self.label1 = QLabel(self)
        self.label1.move(200, 705)
        self.label1.resize(300, 20)
        self.label1.setText("")

        self.label2 = QLabel(self)
        self.label2.move(430, 20)
        self.label2.resize(300, 20)
        self.label2.setText("Insert model description")

        # Create textbox
        self.textbox = QPlainTextEdit(self)
        self.textbox.move(20, 60)
        self.textbox.resize(1000, 600)

        # Create a button in the window
        self.evalButton = QPushButton('Evaluate', self)
        self.evalButton.move(850, 700)

        # Create a button in the window
        self.loadButton = QPushButton('Load Dataset', self)
        self.loadButton.move(80, 700)

        self.chooseButton = QPushButton('Choose algorithms', self)
        self.chooseButton.resize(150, 30)
        self.chooseButton.move(400, 700)

        self.algorithmsNames = ['Lasso', 'Ridge', 'RandomForest', 'RFECV_SVM']
        self.algorithmsCheckboxesModel = QStandardItemModel()
        for i, name in enumerate(self.algorithmsNames):
            item = QStandardItem(name)
            item.setCheckState(False)
            item.setCheckable(True)
            self.algorithmsCheckboxesModel.appendRow(item)

        # closeButton = QPushButton('Close', self.algorithmsCheckboxesModel)
        # self.algorithmsCheckboxes.appendRow(closeButton)

        self.view = QListView()
        self.view.setWindowTitle('algorithms')
        self.view.setModel(self.algorithmsCheckboxesModel)

        # connect button to function on_click
        self.evalButton.clicked.connect(self.evaluateOnClick)
        self.loadButton.clicked.connect(self.openFileNameDialog)
        self.chooseButton.clicked.connect(self.proceedalgorithmsChoice)

        self.show()

    def proceedalgorithmsChoice(self):
        self.view.show()

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;CSV Files (*.csv)", options=options)
        if fileName:
            self.label1.setText(fileName)

    @pyqtSlot()
    def evaluateOnClick(self):
        q_model = self.view.model()
        labels = self.getSelectedItemsLabels(q_model)
        # model = Model()
        if len(labels) > 0:
            models = []
            for label in labels:
                if label == 'Lasso':
                    models.append(LogisticRegressionCV(penalty='l2', solver='newton-cg', multi_class='multinomial'))
                elif label == 'Ridge':
                    models.append(LogisticRegressionCV(penalty='l1', solver='liblinear'))
                elif label == 'RandomForest':
                    models.append(RandomForestClassifier(n_estimators=100))
                elif label == 'RFECV_SVM':
                    models.append(RFECV(estimator=SVC(gamma="scale", kernel="linear"), verbose=1))
            model = Model(models)
            # print(model, models)
        else:
            plain_text = self.textbox.toPlainText()
            json_components = json.loads(plain_text)
            model = Model(json_components)
            model = model.from_json(json_components)
            # print("model", model, json_components, sep="\n")
        data = self.getDataFromFile(self.label1.text())
        training_size = len(data[0]) // 2
        model = self.getTrainedAndValidatedModel(model, data, training_size)
        feature_ranking = model.feature_ranking()
        performed_voting = model.perform_voting()
        QMessageBox.question(self, "Genomics Studies - Summary",
                             "\n Performed voting: \n" +
                             "\n".join(["Feature " + self.getPretty(i) + " : " + str(performed_voting[i]) for i in range(len(performed_voting))]),
                             QMessageBox.Ok, QMessageBox.Ok)

    def getDataFromFile(self, path=""):
        if not path:
            return data_sim()
        else:
            with open(path) as file:
                input_data_lines = file.read().splitlines()
                n = len(input_data_lines)
                m = len(re.split("\\s*;\\s*", input_data_lines[0]))
                data = (np.ndarray((n, m-1)), np.ndarray((n, )))
                for i, line in enumerate(input_data_lines):
                    row = re.split("\\s*;\\s*", line)
                    numbers = [float(x) for x in row[:-1]]
                    data[0][i] = np.array(numbers)
                    data[1][i] = int(row[-1])
                return data

    def getSelectedItemsLabels(self, model: QStandardItemModel):
        labels = []
        for row in range(model.rowCount()):
            item = model.item(row)
            if item.checkState() == Qt.Checked:
                labels.append(item.text())
        return labels

    def getTrainedAndValidatedModel(self, model: Model, data, training_size, validator: Validator = Validator()):
        X, y = data
        X_tr = X[:training_size, :]
        y_tr = y[:training_size]
        X_test = X[training_size:, :]
        y_test = y[training_size:]
        # print(X_tr, y_tr, X_test, y_test, sep="\n")
        model.fit(X_tr, y_tr)
        model.set_validation(validator)
        model.validate(X_test, y_test)
        return model

    def getPretty(self, i, char_num = 3):
        s = str(i)
        return "".join([" " for x in range(char_num - len(s))]) + s



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())