# main.py
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager
from kivy.lang import Builder

from trainpage import trainpage
from testpage import testpage
from resultpage import resultpage

Builder.load_file('trainpage/trainpage.kv')
Builder.load_file('testpage/testpage.kv')
Builder.load_file('resultpage/resultpage.kv')
class MyApp(App):
    def build(self):
        self.sm = ScreenManager()
        self.sm.add_widget(trainpage.TrainPage(name='train'))
        self.sm.add_widget(testpage.TestPage(name='test'))
        self.sm.add_widget(resultpage.ResultPage(name='result'))
        self.sm.current = 'train'
        self.sm.current = 'test'
        self.sm.current = 'result'
        self.sm.current = 'train'
        return self.sm

    def change_train_page(self):
        self.sm.current = 'train'

    def change_test_page(self):
        self.sm.current = 'test'

    def change_result_page(self):
        self.sm.current = 'result'

if __name__ == '__main__':
    MyApp().run()
