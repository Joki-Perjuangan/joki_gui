from kivy.app import App
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.properties import ListProperty

from skimage import io, color, feature, transform
import joblib

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog

class TestPage(Screen):
    selected_file = ''
    benar = 0
    salah = 0
    predict = 0

    def select_image(self):
        root = tk.Tk()
        root.withdraw()  # Hide the main window

        file_selected = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif")]
        )
        root.destroy()  # Destroy the Tkinter window to avoid memory leaks

        if file_selected:
            self.ids.image_path.text = f"Selected Image: {file_selected}"[:30] + '...'
            self.ids.image_test.source = file_selected
            self.selected_file = file_selected

    def on_spinner_select(self, spinner, text):
        self.choose = self.options.index(text)

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

    def compute_nilai(self, k, target_size=(128, 128), lbp_radius=1, lbp_n_points=8,
                                             hog_orientations=9, hog_pixels_per_cell=(8, 8),
                                             hog_cells_per_block=(2, 2)):
        lbp_features_list = []
        hog_features_list = []

        image = io.imread(self.selected_file)
        if len(image.shape) == 3:
            image = color.rgb2gray(image)

        # Resize image
        image_resized = transform.resize(image, target_size)

        # Compute LBP
        lbp_features = self.compute_lbp(image_resized)
        lbp_features_list.append(lbp_features.flatten())
        # Compute HOG
        hog_features = self.compute_hog(image_resized)
        hog_features_list.append(hog_features.flatten())
        return lbp_features_list, hog_features_list

    def test(self):
        input_text = self.ids.input_box.text
        input_value = int(input_text)
        lbp_features, hog_features = self.compute_nilai(input_value)
        lbp_features = np.array(lbp_features)
        concatenated_features = np.concatenate((lbp_features, hog_features), axis=1)

        knn_model_file = "./KNN_Model.pkl"
        scaler_file = "./scaler.pkl"
        knn = joblib.load(knn_model_file)
        scaler = joblib.load(scaler_file)

        scaled_data = scaler.transform(concatenated_features)
        prediction = knn.predict_proba(scaled_data)

        label_class = list(pd.read_csv('./labelling.csv')['0'])

        self.predict = prediction[0]
        idx = np.argmax(self.predict)

        print(self.predict)

        if max(prediction[0]) > 0.99:
            self.ids.nama_tanaman.text = label_class[idx]
            self.ids.akurasi.text = "{:.5g}".format(max(prediction[0]) * 100) + '%'
            self.benar += 1
        elif max(prediction[0]) > 0.5:
            self.ids.nama_tanaman.text = label_class[idx]
            self.ids.akurasi.text = "{:.5g}".format(max(prediction[0]) * 100) + '%'
            self.salah += 1
        else:
            self.ids.akurasi.text = 'Tidak Sesuai'
            self.ids.nama_tanaman.text = ''
            self.salah += 1

        self.ids.Nilai_hog.text = 'Nilai Hog : ' + str(hog_features[0])[1:10] + '..'
        self.ids.Nilai_lbp.text = 'Nilai Lbp : ' + str(lbp_features[0])[1:15] + '..'

class TestPageApp(App):
    def build(self):
        self.sm = ScreenManager()
        self.sm.add_widget(TestPage(name='home'))
        self.sm.current = 'home'
        return self.sm

if __name__ == '__main__':
    TestPageApp().run()
