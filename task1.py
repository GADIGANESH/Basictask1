import sys
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                             QWidget, QListWidget, QLineEdit, QLabel, QHBoxLayout, QMessageBox)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TaskManagerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initDataFrame()
        self.initUI()

    def initDataFrame(self):
        try:
            self.tasks = pd.read_excel('tasks.xlsx', engine='openpyxl')
        except FileNotFoundError:
            self.tasks = pd.DataFrame(columns=['description', 'priority'])

    def initUI(self):
        self.setWindowTitle("Task Manager")
        self.setGeometry(100, 100, 600, 400)

        mainLayout = QVBoxLayout()

        self.listWidget = QListWidget()
        mainLayout.addWidget(self.listWidget)

        # Description input
        descriptionLayout = QHBoxLayout()
        descriptionLabel = QLabel("Description:")
        self.descriptionInput = QLineEdit()
        descriptionLayout.addWidget(descriptionLabel)
        descriptionLayout.addWidget(self.descriptionInput)

        # Priority input
        priorityLayout = QHBoxLayout()
        priorityLabel = QLabel("Priority:")
        self.priorityInput = QLineEdit()
        priorityLayout.addWidget(priorityLabel)
        priorityLayout.addWidget(self.priorityInput)

        mainLayout.addLayout(descriptionLayout)
        mainLayout.addLayout(priorityLayout)

        addButton = QPushButton("Add Task")
        addButton.clicked.connect(self.addTask)
        mainLayout.addWidget(addButton)

        removeButton = QPushButton("Remove Selected Task")
        removeButton.clicked.connect(self.removeTask)
        mainLayout.addWidget(removeButton)

        recommendButton = QPushButton("Recommend Similar Task")
        recommendButton.clicked.connect(self.recommendTask)
        mainLayout.addWidget(recommendButton)

        self.updateListWidget()

        centralWidget = QWidget()
        centralWidget.setLayout(mainLayout)
        self.setCentralWidget(centralWidget)

    def addTask(self):
        description = self.descriptionInput.text()
        priority = self.priorityInput.text()
        if description and priority:
            new_id = self.tasks['id'].max() + 1 if not self.tasks.empty else 1
            new_task = pd.DataFrame({'id': [new_id], 'description': [description], 'priority': [priority]})
            self.tasks = pd.concat([self.tasks, new_task], ignore_index=True)
            self.updateListWidget()
            self.saveTasks()

    def removeTask(self):
        selectedItems = self.listWidget.selectedItems()
        if not selectedItems:
            QMessageBox.warning(self, "Warning", "Please select a task to remove.")
            return
        for item in selectedItems:
            idx = self.listWidget.row(item)
            self.tasks.drop(idx, inplace=True)
        self.tasks.reset_index(drop=True, inplace=True)
        self.saveTasks()
        self.updateListWidget()

    def recommendTask(self):
        selectedItems = self.listWidget.selectedItems()
        if not selectedItems or len(self.tasks) < 2:
            QMessageBox.warning(self, "Warning", "Select a task for recommendations.")
            return
        selectedText = selectedItems[0].text()
        description = selectedText.split(" - Priority: ")[0]

        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.tasks['description'])
        cosine_sim = cosine_similarity(tfidf_matrix)

        selected_idx = self.listWidget.currentRow()
        sim_scores = list(enumerate(cosine_sim[selected_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:4]

        msg = "Recommended Tasks based on description similarity:\n"
        for i, (idx, score) in enumerate(sim_scores, 1):
            if score > 0.1:  # Adjust this threshold as necessary
                msg += f"{i}: {self.tasks.iloc[idx]['description']} (Similarity: {score:.2f})\n"

        QMessageBox.information(self, "Recommendations", msg)

    def saveTasks(self):
        self.tasks.to_excel('tasks.xlsx', index=False, engine='openpyxl')

    def updateListWidget(self):
        self.listWidget.clear()
        for _, task in self.tasks.iterrows():
            self.listWidget.addItem(f"{task['description']} - Priority: {task['priority']}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = TaskManagerApp()
    ex.show()
    sys.exit(app.exec_())
