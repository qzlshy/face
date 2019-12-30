#!/usr/bin/python3
# coding=utf-8

import cv2
from managers import WindowsManager, CaptureManager
import face_recognition
import numpy as np


class Cameo(object):
    def __init__(self):
        self._windowManager = WindowsManager('Cameo', self.onkeypress)
        self._capturemanager = CaptureManager(cv2.VideoCapture(0), self._windowManager, True)

    def run(self):
        """run the main loop"""
        face_cascade=cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
        self._windowManager.create_window()
        while self._windowManager.is_window_created:
            self._capturemanager.enterframe()
            frame = self._capturemanager.frame
            # 这里插入滤波代码
            
            self._capturemanager.exitframe()
            self._windowManager.process_events()

    def onkeypress(self, keycode):
        """处理按键操作
        空格 表示 截图
        tab 表示 开始/停止 记录 screencast
        escape 表示退出
        """

        if keycode == 32: # 空格
            self._capturemanager.face_code()
        elif keycode == 27:  # escape 键
            self._capturemanager.save_face_code('./face_code.npy')
            self._windowManager.destroy_window()
            print("正在退出")


if __name__ == "__main__":
    Cameo().run()
