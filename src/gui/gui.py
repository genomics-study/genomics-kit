import re
import sys

import numpy as np
import pyqtgraph as pg
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

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.label1 = QLabel(self)
        self.label1.move(200, 705)
        self.label1.resize(300, 20)
        self.label1.setText("")

        self.label2 = QLabel(self)
        self.label2.move(430, 20)
        self.label2.resize(300, 20)
        self.label2.setText("Insert model description")
        self.label2.hide()

        self.textbox = QPlainTextEdit(self)
        self.textbox.move(20, 60)
        self.textbox.resize(1000, 600)
        self.textbox.hide()

        self.textbox2 = QPlainTextEdit(self)        # Unused textbox
        self.textbox2.move(20, 60)
        self.textbox2.resize(1000, 600)
        self.textbox2.setReadOnly(True)

        self.load_dataset_button = QPushButton('Load Dataset', self)
        self.load_dataset_button.move(80, 700)

        self.components_choice_button = QPushButton('Choose components', self)
        self.components_choice_button.resize(150, 30)
        self.components_choice_button.move(440, 700)

        self.evaluate_button = QPushButton('Evaluate', self)
        self.evaluate_button.move(850, 700)

        self.components_choice_window = ComponentsChoiceWindow(self)

        self.algorithms_names = ['Lasso', 'Ridge', 'RandomForest', 'RFECV_SVM']
        self.algorithms_checkboxes_model = QStandardItemModel()
        for i, name in enumerate(self.algorithms_names):
            item = QStandardItem(name)
            item.setCheckState(False)
            item.setCheckable(True)
            self.algorithms_checkboxes_model.appendRow(item)

        self.algorithms_view = QListView()
        self.algorithms_view.setWindowTitle('Algorithms')
        self.algorithms_view.setModel(self.algorithms_checkboxes_model)

        # connect button to function on_click
        self.evaluate_button.clicked.connect(self.evaluateOnClick)
        self.load_dataset_button.clicked.connect(self.openFileNameDialog)
        self.components_choice_button.clicked.connect(self.proceedComponentsChoice)

        self.show()

    def proceedComponentsChoice(self):
        self.components_choice_window.showNormal()

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Choose dataset file", "",
                                                  "All Files (*);;CSV Files (*.csv)", options=options)
        if file_name:
            self.label1.setText(file_name)

    def proceedAlgorithmsChoice(self):
        self.algorithms_view.show()

    @pyqtSlot()
    def evaluateOnClick(self):
        q_model = self.algorithms_view.model()
        labels = self.getSelectedItemsLabels(q_model)
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
        else:
            plain_text = self.textbox.toPlainText()
            json_components = json.loads(plain_text)
            model = Model(json_components)
            model = model.from_json(json_components)
        data = self.getDataFromFile(self.label1.text())
        training_size = len(data[0]) // 2
        model, validation = self.getTrainedAndValidatedModelWithValidation(model, data, training_size)
        feature_ranking = model.feature_ranking()
        voting_results = model.perform_voting()
        QMessageBox.question(self, "Genomics Studies - Summary",
                             "\n Voting results: \n" +
                             "\n".join(["Feature " + self.getPretty(i) + " : " + str(v) for (i, v) in enumerate(voting_results)]),
                             QMessageBox.Ok, QMessageBox.Ok)

        sorted_voting = [(i, v) for (v, i) in reversed(sorted([(voting_results[i], i) for i in range(len(voting_results))]))]
        QMessageBox.question(self, "Genomics Studies - Summary",
                             "\n Features sorted by votes: \n" +
                             "\n".join(["Feature " + self.getPretty(i) + " : " + str(v) for (i, v) in sorted_voting]),
                             QMessageBox.Ok, QMessageBox.Ok)
        # print(feature_ranking, "\n".join([str(key) for key in feature_ranking]), sep="\n")
        self.writeResultToFile(sorted_voting, suggested_name="voting_results.csv",
                               first_row="Feature; Voting result",
                               command="Choose output file for voting results")
        self.writeResultToFile(validation, suggested_name="validation_results.txt",
                               command="Choose output file for validation results")
        self.writeResultToFile(feature_ranking, suggested_name="feature_ranking.txt",
                               command="Choose output file for feature ranking")
        results_chart_window = ResultsChartWindow(self, model)

    def writeResultToFile(self, result, suggested_name="", first_row="", command="Choose output file"):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(self, command, suggested_name,
                                                   "All Files (*);;CSV Files (*.csv)", options=options)
        if file_name:
            with open(file_name, "w+") as file:
                if first_row:
                    file.write(str(first_row) + "\n")
                file.write("\n".join([str(i) + "; " + str(v) for (i, v) in enumerate(result)]))

    def parseLine(self, line):
        row = re.split("[\\s;,]*", line)
        if row[0] == "":
            row = row[1:]
        if row[-1] == "":
            row = row[:-1]
        return row

    def getDataFromFile(self, path=""):
        if not path:
            return data_sim()
        else:
            with open(path) as file:
                input_data_lines = file.read().splitlines()
                n = len(input_data_lines)
                m = len(self.parseLine(input_data_lines[0]))
                data = (np.ndarray((n, m-1)), np.ndarray((n, )))
                for i, line in enumerate(input_data_lines):
                    row = self.parseLine(line)
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

    def getTrainedAndValidatedModelWithValidation(self, model: Model, data, training_size, validator: Validator = Validator()):
        X, y = data
        X_tr = X[:training_size, :]
        y_tr = y[:training_size]
        X_test = X[training_size:, :]
        y_test = y[training_size:]
        model.fit(X_tr, y_tr)
        model.set_validation(validator)
        validation = model.validate(X_test, y_test)
        return model, validation

    def getPretty(self, i, char_num=3):
        s = str(i)
        return "".join([" " for x in range(char_num - len(s))]) + s


class ResultsChartWindow(QMainWindow):
    def __init__(self, app, model):
        super().__init__()
        self.width = 800
        self.height = 640
        self.left = 10
        self.top = 10
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setWindowTitle("Genomics Studies - Summary")
        self.app = app
        self.model = model
        self.voting_results = model.perform_voting()
        self.voting_results_window = pg.plot()

        self.voting_results_window.setWindowTitle('Voting Results')
        x, y = list(zip(*enumerate(self.voting_results)))
        voting_results_graph = pg.BarGraphItem(x=x, height=y, width=0.6, brush='y')
        self.voting_results_window.addItem(voting_results_graph)


class ComponentsChoiceWindow(QMainWindow):
    def __init__(self, app):
        super().__init__()
        self.width = 800
        self.height = 640
        self.left = 10
        self.top = 10
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setWindowTitle("Choose components")
        self.app = app

        # self.option_names = ['Load json', 'Paste json', 'Choose from defaults']
        # self.option_checkboxes_model = QStandardItemModel()
        #
        # for i, name in enumerate(self.option_names):
        #     item = QStandardItem(name)
        #     item.setCheckState(False)
        #     item.setCheckable(True)
        #     self.option_checkboxes_model.appendRow(item)
        #
        # self.option_view = QListView()
        # self.option_view.setWindowTitle('Options')
        # self.option_view.setModel(self.option_checkboxes_model)
        #
        # # self.optionView.showNormal()
        #
        # self.central_widget = QWidget()
        # self.setCentralWidget(self.central_widget)
        #
        # self.layout = QVBoxLayout(self.central_widget)
        # self.setLayout(self.layout)
        # self.layout.addWidget(self.option_view)

        self.choose_algorithms_button = QPushButton('Choose algorithms', self)
        self.choose_algorithms_button.resize(150, 30)
        self.choose_algorithms_button.move(400, 400)
        self.choose_algorithms_button.clicked.connect(self.proceedAlgorithmsChoice)

        self.load_json_button = QPushButton('Paste json model', self)
        self.load_json_button.resize(150, 30)
        self.load_json_button.move(250, 400)
        self.load_json_button.clicked.connect(self.proceedLoadJson)

    def proceedAlgorithmsChoice(self):
        self.app.algorithms_view.show()

    def proceedLoadJson(self):
        load_json_window = LoadJsonWindow(self)
        load_json_window.show()
        self.app.textbox = self.textbox


class LoadJsonWindow(QMainWindow):
    def __init__(self, app):
        super().__init__()
        self.width = 800
        self.height = 640
        self.left = 10
        self.top = 10
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setWindowTitle("Paste json model")
        self.app = app

        self.textbox = QPlainTextEdit(self)
        self.textbox.move(20, 60)
        self.textbox.resize(1000, 600)
        self.textbox.showNormal()
        self.app.textbox = self.textbox


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
