from kivy.app import App
from kivy.uix.screenmanager import Screen, ScreenManager

import numpy as np
import pandas as pd
class ResultPage(Screen):
    acc = 0

    def show(self):
        predict = self.manager.get_screen('test').predict

        benar = self.manager.get_screen('test').benar
        salah = self.manager.get_screen('test').salah

        self.acc = benar / (benar + salah)
        self.ids.akurasi.text = "{:.5g}".format(self.acc * 100) + '%'

        idx = np.argmax(predict)

        self.ids.Nilai_Lbp.text = str(list(pd.read_csv('./avg_lbp.csv').iloc[idx]))[1:10]
        self.ids.Nilai_Hog.text = str(list(pd.read_csv('./avg_hog.csv').iloc[idx]))[1:15]

        self.ids.nama_tanaman.text = list(pd.read_csv('./labelling.csv')['0'])[idx]

    def reset(self):
        self.manager.get_screen('train').ids.folder_path.text = 'Menunggu folder input'
        self.manager.get_screen('train').ids.input_box.text = ''
        self.manager.get_screen('train').selected_folder = ""
        self.manager.get_screen('train').labels = []
        self.manager.get_screen('train').lbp_features_list = []
        self.manager.get_screen('train').hog_features_list = []
        self.manager.get_screen('train').label_class = []

        self.manager.get_screen('test').ids.Nilai_hog.text = 'Nilai Hog :'
        self.manager.get_screen('test').ids.Nilai_lbp.text = 'Nilai Lbp :'
        self.manager.get_screen('test').ids.akurasi.text = 'menunggu data'
        self.manager.get_screen('test').ids.nama_tanaman.text = 'Menunggu data'
        self.manager.get_screen('test').ids.image_test.source = 'no_image.png'
        self.manager.get_screen('test').ids.image_path.text = 'Menunggu image input'
        # self.manager.get_screen('test').ids.my_spinner.text = 'Select an option'
        self.manager.get_screen('test').ids.input_box.text = ''
        self.manager.get_screen('test').selected_file = ''
        self.manager.get_screen('test').benar = 0
        self.manager.get_screen('test').salah = 0
        # self.manager.get_screen('test').options = ['Option 1', 'Option 2', 'Option 3']
        # self.manager.get_screen('test').choose = 0
        self.manager.get_screen('test').predict = 0

        self.acc = 0

        self.ids.akurasi.text = 'Menunggu data'
        self.ids.Nilai_Lbp.text = 'Menunggu data'
        self.ids.Nilai_Hog.text = 'Menunggu data'
        self.ids.nama_tanaman.text = 'Menunggu data'

        self.manager.current = 'train'

class ResultPageApp(App):
    def build(self):
        self.sm = ScreenManager()
        self.sm.add_widget(ResultPage(name='home'))
        self.sm.current = 'home'
        return self.sm

if __name__ == '__main__':
    ResultPageApp().run()
