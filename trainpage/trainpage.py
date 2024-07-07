from kivy.app import App
from kivy.uix.screenmanager import Screen, ScreenManager

import tkinter as tk
from tkinter import filedialog

import csv
import numpy as np
from skimage import io, color, feature, transform
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier

import os
import joblib
from datetime import datetime
import concurrent.futures

import pandas as pd


class FuzzyKNN:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.knn = NearestNeighbors(n_neighbors=n_neighbors)
        self.train_data = None
        self.train_labels = None
        self.classes = None

    def fit(self, X, y):
        self.train_data = X
        self.train_labels = y
        self.classes = np.unique(y)
        self.knn.fit(X)

    def _compute_membership_values(self, distances, indices):
        num_classes = len(self.classes)
        membership_values = np.zeros(num_classes)

        for idx, distance in zip(indices, distances):
            neighbor_label = self.train_labels[idx]
            membership_value = 1 / (distance + 1e-5)
            class_index = np.where(self.classes == neighbor_label)[0][0]
            membership_values[class_index] += membership_value

        membership_values /= membership_values.sum()
        return membership_values

    def predict(self, X, k=None):
        if k is None:
            k = self.n_neighbors

        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(self.train_data)
        distances, indices = knn.kneighbors(X)
        predictions = []

        for i in range(X.shape[0]):
            membership_values = self._compute_membership_values(distances[i], indices[i])
            predicted_class = self.classes[np.argmax(membership_values)]
            predictions.append(predicted_class)

        return np.array(predictions)

    def predict_proba(self, X, k=None):
        if k is None:
            k = self.n_neighbors

        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(self.train_data)
        distances, indices = knn.kneighbors(X)
        probabilities = []

        for i in range(X.shape[0]):
            membership_values = self._compute_membership_values(distances[i], indices[i])
            probabilities.append(membership_values)

        return np.array(probabilities)

class TrainPage(Screen):
    selected_folder = ""
    labels = []
    lbp_features_list = []
    hog_features_list = []
    label_class = []
    lbp_shape = 0
    hog_shape = 0
    target_size = (128, 128)
    k = 0

    def select_folder(self):
        root = tk.Tk()
        root.withdraw()  # Hide the main window

        folder_selected = filedialog.askdirectory()
        root.destroy()  # Destroy the Tkinter window to avoid memory leaks

        if folder_selected:
            self.ids.folder_path.text = f"Selected Folder: {folder_selected}"
            self.selected_folder = folder_selected

    def compute_lbp(self, image, radius=1, n_points=8):
        # Compute LBP
        lbp = feature.local_binary_pattern(image, n_points, radius, method='uniform')
        return lbp

    def compute_hog(self, image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        # Compute HOG
        hog_features = feature.hog(image,
                                   orientations=orientations,
                                   pixels_per_cell=pixels_per_cell,
                                   cells_per_block=cells_per_block,
                                   block_norm='L2-Hys')
        return hog_features

    def compute_lbp_and_hog_batch_from_paths(self):
        lbp_features_list = []
        hog_features_list = []

        avg_lbp = []
        avg_hog = []

        self.label_class = os.listdir(self.selected_folder)

        for i, img_dirs in enumerate(os.listdir(self.selected_folder)):
            img_dir = os.path.join(self.selected_folder, img_dirs)
            temp_lbp = []
            temp_hog = []
            for img_filename in os.listdir(img_dir):
                img_path = os.path.join(img_dir, img_filename)

                image = io.imread(img_path)
                if len(image.shape) == 3:
                    image = color.rgb2gray(image)

                image_resized = transform.resize(image, self.target_size)

                lbp_features = self.compute_lbp(image_resized)
                temp_lbp.append(lbp_features.flatten())

                hog_features = self.compute_hog(image_resized)
                temp_hog.append(hog_features.flatten())

                self.labels.append(i)

            lbp_features_list += temp_lbp
            hog_features_list += temp_hog

            mean_lbp = np.mean(temp_lbp, axis=0)
            mean_hog = np.mean(temp_hog, axis=0)

            avg_lbp.append(mean_lbp)
            avg_hog.append(mean_hog)

            temp_lbp = []
            temp_hog = []

        avg_lbp_df = pd.DataFrame(avg_lbp)
        avg_hog_df = pd.DataFrame(avg_hog)

        avg_lbp_df.to_csv('./avg_lbp.csv', index=False)
        avg_hog_df.to_csv('./avg_hog.csv', index=False)

        self.lbp_features_list = lbp_features_list
        self.hog_features_list = hog_features_list

        return lbp_features_list, hog_features_list

    def train(self):
        input_text = self.ids.input_box.text
        input_value = int(input_text)
        lbp_features, hog_features = self.compute_lbp_and_hog_batch_from_paths()
        lbp_features = np.array(lbp_features)
        lbp_data = pd.DataFrame(lbp_features)

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')[:-3]
        lbp_data.to_csv('./lbp_features.csv', index=False)
        lbp_data.to_csv(f'./csvresult/lbp_{timestamp}.csv', index=False)
        hog_data = pd.DataFrame(hog_features)
        hog_data.to_csv('./hog_features.csv', index=False)
        hog_data.to_csv(f'./csvresult/hog_{timestamp}.csv', index=False)

        concatenated_features = np.concatenate((lbp_features, hog_features), axis=1)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(concatenated_features)
        knn_classifier = KNeighborsClassifier(n_neighbors=input_value)
        knn_classifier.fit(scaled_features, self.labels)

        joblib_file = "./KNN_Model.pkl"
        joblib.dump(knn_classifier, joblib_file)
        scaler_file = "./scaler.pkl"
        joblib.dump(scaler, scaler_file)

        label_df = pd.DataFrame(self.label_class)
        label_df.to_csv('./labelling.csv', index= False)

        self.manager.get_screen('test').options = self.label_class
          # Predicting the first image's features
        return knn_classifier, scaler
class TrainPageApp(App):
    def build(self):
        self.sm = ScreenManager()
        self.sm.add_widget(TrainPage(name='home'))
        self.sm.current = 'home'
        return self.sm

if __name__ == '__main__':
    TrainPageApp().run()
