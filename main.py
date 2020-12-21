from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm import tqdm

from faceswapui import Ui_Dialog
import os
import sys
import dlib

import models
import NonLinearLeastSquares
import ImageProcessing

from drawing import *

import FaceRendering
import utils

from time import sleep
import cv2
import sys
from threading import Timer, Thread, Event
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QVBoxLayout
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import QThread, pyqtSlot
from PyQt5.QtWidgets import *
from PyQt5.QtCore import (QDir)

import numpy as np

from weakref import ref
import pynput

if not os.path.exists("./output/"):
    os.makedirs("./output/")


class perpetualTimer:

    def __init__(self, t, hFunction):
        self.t = t
        self.hFunction = hFunction
        self.thread = Timer(self.t, self.handle_function)

    def handle_function(self):
        self.hFunction()
        self.thread = Timer(self.t, self.handle_function)
        self.thread.start()

    def start(self):
        self.thread.start()

    def cancel(self):
        self.thread.cancel()


class Communicate(QtCore.QObject):
    """
    Small Class based on a Widget
    It's only purpose is to contain a signal object
    Note: Type QObject
    """
    sig = QtCore.pyqtSignal(object)


class workerThread(QThread):
    updatedM = QtCore.pyqtSignal(int)

    def __init__(self, mw):
        self.mw = mw
        QThread.__init__(self)

    def __del__(self):
        self.wait()

    def run(self):

        itr = 0
        QApplication.processEvents()

        while self.mw.isRun:
            itr += 1

            if self.mw.isthreadActive and not self.mw.isbusy and self.mw.frameID != self.mw.cap.get(
                    cv2.CAP_PROP_POS_FRAMES):

                if np.abs(self.mw.frameID - self.mw.cap.get(cv2.CAP_PROP_POS_FRAMES)) > 1:
                    self.mw.cap.set(cv2.CAP_PROP_POS_FRAMES, self.mw.frameID)

                if self.mw.timer is None:
                    self.mw.frameID += 1

                self.mw.isbusy = True

                ret, image = self.mw.cap.read()
                self.mw.limg = image

                if not ret:
                    self.mw.isthreadActiv = False
                    self.mw.isbusy = False
                    continue

                nchannel = image.shape[2]
                limg2 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                timg = cv2.resize(limg2, (int(limg2.shape[1]), int(limg2.shape[0])))
                limage = QtGui.QImage(timg.data, timg.shape[1], timg.shape[0], nchannel * timg.shape[1],
                                      QtGui.QImage.Format_RGB888)

                self.mw.OriginVideoLB.setPixmap(QtGui.QPixmap(limage))
                self.mw.OriginVideoLB.setAlignment(QtCore.Qt.AlignCenter)
                self.mw.OriginVideoLB.setScaledContents(True)
                self.mw.OriginVideoLB.setMinimumSize(1, 1)

                if self.mw.isProceed:
                    if np.abs(self.mw.frameID - self.mw.procCap.get(cv2.CAP_PROP_POS_FRAMES)) > 1:
                        self.mw.procCap.set(cv2.CAP_PROP_POS_FRAMES, self.mw.frameID)

                    ret, image = self.mw.procCap.read()
                    self.mw.limg = image

                    if not ret:
                        self.mw.isthreadActiv = False
                        self.mw.isbusy = False
                        continue

                    nchannel = image.shape[2]
                    limg2 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    timg = cv2.resize(limg2, (int(limg2.shape[1]), int(limg2.shape[0])))
                    limage = QtGui.QImage(timg.data, timg.shape[1], timg.shape[0], nchannel * timg.shape[1],
                                          QtGui.QImage.Format_RGB888)

                    self.mw.ProceedVideoLB.setPixmap(QtGui.QPixmap(limage))
                    self.mw.ProceedVideoLB.setAlignment(QtCore.Qt.AlignCenter)
                    self.mw.ProceedVideoLB.setScaledContents(True)
                    self.mw.ProceedVideoLB.setMinimumSize(1, 1)

                if not self.mw.sliderbusy and not self.mw.sliderbusy2:
                    self.updatedM.emit(self.mw.frameID)

                QApplication.processEvents()
                self.mw.isbusy = False
            else:
                if self.mw.isthreadActive and self.mw.timer is None:
                    self.mw.frameID += 1
                sleep(1.0 / 50)


class main(Ui_Dialog):

    def __init__(self, *argv, **argvw):
        super(self.__class__, self).__init__()
        # self.c = Communicate()

        self.frameID = 0
        self.fps = 1.0
        self.stframe = 0

        self.isRun = True
        self.isthreadActive = False
        self.sliderbusy = False
        self.sliderbusy2 = False

        self.wthread = workerThread(self)
        self.wthread.updatedM.connect(self.horizontalSliderSet)
        self.wthread.start()

        self.isbusy = 0
        self.frameHeight = 1
        self.frameWidth = 1
        self.limg = np.zeros((1, 1, 1))
        self.cap = None
        self.timer = None

        self.isProceed = False
        self.procCap = None

    def on_release(self, key):
        if self.isRun:
            if key == pynput.keyboard.Key.left:
                # print('left')
                self.horizontalSliderIncrease(15)
            elif key == pynput.keyboard.Key.right:
                # print('right')
                self.horizontalSliderIncrease(-15)
            elif pynput.keyboard.Key.space == key:
                if self.pauseButton.isEnabled():
                    self.pauseButtonPressed()

    def on_press(self, key):
        if self.isRun:
            if pynput.keyboard.Key.space == key:
                if self.pauseButton.isEnabled():
                    self.pauseButtonPressed()
                else:
                    while (self.sliderbusy == True):
                        sleep(0.1)
                    self.startButtonPressed()

    def setupUi(self, *argv, **argvw):
        super(main, self).setupUi(*argv, **argvw)

        self.VideoFileBtn.clicked.connect(self.OpenVideoFile)
        self.FaceFileBtn.clicked.connect(self.OpenFaceImageFile)
        self.RunBtn.clicked.connect(self.SwapFace)
        self.startButton.clicked.connect(self.startButtonPressed)
        self.pauseButton.clicked.connect(self.pauseButtonPressed)
        self.compareButton.clicked.connect(self.startButtonPressed)
        self.ExitBtn.clicked.connect(self.exitButtonPressed)

        self.horizontalSlider.sliderPressed.connect(self.horizontalSliderPressed)
        self.horizontalSlider.sliderReleased.connect(self.horizontalSliderReleased)
        self.horizontalSlider.valueChanged.connect(self.slider_value_changed)

        klistener = pynput.keyboard.Listener(on_press=self.on_release)  # ,on_release=self.on_release)
        klistener.start()

    def OpenVideoFile(self):
        self.progressBar.setValue(0)
        fileName, fiter = QFileDialog.getOpenFileName(
            None, 'Open Video File', os.getcwd(), 'Video Files (*.mp4 *.avi)')

        if fileName:
            self.VideoFileNameLE.setText(fileName)
            self.startButton.setEnabled(True)
            self.horizontalSlider.setEnabled(True)
            # self.DisplayVideo(fileName)

            self.cap = cv2.VideoCapture(fileName)
            self.isvideo = True

            length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.stframe = int(0 * self.fps)
            self.endframe = int(length * self.fps)

            self.horizontalSlider.setMaximum(length - 1)
            # self.horizontalSlider.setMaximum(self.endframe - self.stframe)
            self.cap.set(1, self.stframe)
            ret, frame = self.cap.read()
            self.drawmin = 1
            self.frameID = self.stframe
            self.limg = frame
            self.frameHeight = frame.shape[0]
            self.frameWidth = frame.shape[1]

            nchannel = frame.shape[2]
            limg2 = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            timg = cv2.resize(limg2, (int(limg2.shape[1]), int(limg2.shape[0])))
            limage = QtGui.QImage(timg.data, timg.shape[1], timg.shape[0], nchannel * timg.shape[1],
                                  QtGui.QImage.Format_RGB888)

            self.OriginVideoLB.setPixmap(QtGui.QPixmap(limage))
            self.OriginVideoLB.setAlignment(QtCore.Qt.AlignCenter)
            self.OriginVideoLB.setScaledContents(True)
            self.OriginVideoLB.setMinimumSize(1, 1)

            self.statusbar.setText("Ready to Play")

            self.startButton.setEnabled(True)
            self.pauseButton.setEnabled(False)
            self.horizontalSlider.setEnabled(True)

    def OpenFaceImageFile(self):
        self.progressBar.setValue(0)

        fileName, filter = QFileDialog.getOpenFileName(
            None, 'Open Face Image File', os.getcwd(), 'Image Files (*.png *.jpg *.bmp)')

        if fileName:
            self.FaceImageFileNameLE.setText(fileName)
            self.DisplayFaceImage(fileName)

    def DisplayFaceImage(self, fileName):
        input_image = cv2.imread(fileName)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

        nHeight = self.FaceImageLB.height()
        height, width, channels = input_image.shape

        nWidth = int(width * nHeight / height)

        input_image = cv2.resize(input_image, (nWidth, nHeight))
        height, width, channels = input_image.shape
        bytesPerLine = channels * width
        qImg = QtGui.QImage(input_image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap01 = QtGui.QPixmap.fromImage(qImg)
        pixmap_image = QtGui.QPixmap(pixmap01)

        self.FaceImageLB.setPixmap(pixmap_image)
        self.FaceImageLB.setAlignment(QtCore.Qt.AlignCenter)
        # self.FaceImageLB.setScaledContents(True)
        self.FaceImageLB.setMinimumSize(1, 1)

    def process_video(self, in_filename, out_filename, face_filename, keep_audio=True):
        # extract audio clip from src
        if keep_audio == True:
            clip = VideoFileClip(in_filename)
            clip.audio.write_audiofile("./temp/src_audio.mp3", verbose=False)

        predictor_path = "./models/shape_predictor_68_face_landmarks.dat"
        # predictor_path = "./models/shape_predictor_81_face_landmarks.dat"

        # the smaller this value gets the faster the detection will work
        # if it is too small, the user's face might not be detected
        maxImageSizeForDetection = 320

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)

        mean3DShape, blendshapes, mesh, idxs3D, idxs2D = utils.load3DFaceModel("./models/candide.npz")

        projectionModel = models.OrthographicProjectionBlendshapes(blendshapes.shape[0])

        # open source video
        vidcap = cv2.VideoCapture(in_filename)

        # get some parameters from input video
        width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vidcap.get(cv2.CAP_PROP_FPS))
        frames_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        # create a video writer for output
        res_filename = "./output/" + out_filename
        vidwriter = cv2.VideoWriter(res_filename, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                                    (width, height))

        cameraImg = vidcap.read()[1]

        textureImg = cv2.imread(face_filename)
        textureCoords = utils.getFaceTextureCoords(textureImg, mean3DShape, blendshapes, idxs2D, idxs3D, detector,
                                                   predictor)

        renderer = FaceRendering.FaceRenderer(cameraImg, textureImg, textureCoords, mesh)

        destShapes2D = utils.getFaceKeypoints(cameraImg, detector, predictor, maxImageSizeForDetection)
        destShape = destShapes2D[0]
        modelParams = projectionModel.getInitialParameters(mean3DShape[:, idxs3D], destShape[:, idxs2D])

        # 3D model parameter optimization
        modelParams = NonLinearLeastSquares.GaussNewton(modelParams, projectionModel.residual,
                                                        projectionModel.jacobian, (
                                                            [mean3DShape[:, idxs3D],
                                                             blendshapes[:, :, idxs3D]],
                                                            destShape[:, idxs2D]), verbose=0)

        # rendering the model to an image
        destShape3D = utils.getShape3D(mean3DShape, blendshapes, modelParams)

        destRenderedImg = renderer.render(destShape3D)

        self.progressBar.setRange(0, frames_count - 1)

        # iterate over the frames and apply the face swap
        for i in tqdm(range(frames_count - 1)):

            success, cameraImg = vidcap.read()

            self.progressBar.setValue(i + 1)

            if success != True:
                # no frames left => break
                break

            shapes2D = utils.getFaceKeypoints(cameraImg, detector, predictor, maxImageSizeForDetection)
            newImg = cameraImg
            try:

                if shapes2D is not None:
                    for shape2D in shapes2D:
                        # 3D model parameter initialization
                        modelParams = projectionModel.getInitialParameters(mean3DShape[:, idxs3D], shape2D[:, idxs2D])

                        # 3D model parameter optimization
                        modelParams = NonLinearLeastSquares.GaussNewton(modelParams, projectionModel.residual,
                                                                        projectionModel.jacobian, (
                                                                            [mean3DShape[:, idxs3D],
                                                                             blendshapes[:, :, idxs3D]],
                                                                            shape2D[:, idxs2D]), verbose=0)

                        # rendering the model to an image
                        shape3D = utils.getShape3D(mean3DShape, blendshapes, modelParams)
                        renderedImg = renderer.render(shape3D)

                        # blending of the rendered face with the image
                        mask = np.copy(renderedImg[:, :, 0])
                        mask1 = np.copy(destRenderedImg[:, :, 0])
                        renderedImg = ImageProcessing.colorTransfer(cameraImg, renderedImg, mask)
                        # newImg = ImageProcessing.blendImages(renderedImg, cameraImg, mask)
                        newImg = ImageProcessing.blendImages0(renderedImg, cameraImg, mask, mask1)
            except:
                pass

            vidwriter.write(newImg)

        # releas video capture and writer
        vidcap.release()
        vidwriter.release()
        renderer.release()

        # apply audio clip to generated video
        if keep_audio == True:
            video = VideoFileClip("./output/proc_video.avi")
            video.write_videofile(out_filename, audio="./output/src_audio.mp3", progress_bar=False, verbose=False)

    def initButton(self):
        self.VideoFileBtn.setEnabled(False)
        self.FaceFileBtn.setEnabled(False)
        self.startButton.setEnabled(False)
        self.RunBtn.setEnabled(False)
        self.compareButton.setEnabled(False)
        self.pauseButton.setEnabled(False)
        self.statusbar.setEnabled(False)
        self.horizontalSlider.setValue(0)

    def releaseButton(self):
        self.VideoFileBtn.setEnabled(True)
        self.FaceFileBtn.setEnabled(True)
        self.startButton.setEnabled(False)
        self.RunBtn.setEnabled(True)
        self.compareButton.setEnabled(True)
        self.pauseButton.setEnabled(False)
        self.statusbar.setEnabled(True)

    def SwapFace(self):
        videoFileName = self.VideoFileNameLE.text()
        faceFilName = self.FaceImageFileNameLE.text()

        if videoFileName == "" or faceFilName == "":
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
            msg.setText("Video File or Face Image File")
            msg.setInformativeText("Please Select the File")
            msg.setWindowTitle("Error")
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.exec_()
            return

        self.initButton()
        self.isProceed = False

        out_filename = "output.mp4"
        self.process_video(videoFileName, out_filename, faceFilName, False)

        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        msg.setText("Face in video is replaced and the result is ./output/proc_video.avi")
        msg.setInformativeText("Swap Face Success")
        msg.setWindowTitle("Swap Face Result")
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.exec_()

        self.isProceed = True
        self.procCap = None

        res_filename = "./output/" + out_filename
        self.procCap = cv2.VideoCapture(res_filename)

        self.procCap.set(1, self.stframe)
        ret, frame = self.procCap.read()
        self.drawmin = 1
        self.frameID = self.stframe
        self.limg = frame

        nchannel = frame.shape[2]
        limg2 = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        timg = cv2.resize(limg2, (int(limg2.shape[1]), int(limg2.shape[0])))
        limage = QtGui.QImage(timg.data, timg.shape[1], timg.shape[0], nchannel * timg.shape[1],
                              QtGui.QImage.Format_RGB888)

        self.ProceedVideoLB.setPixmap(QtGui.QPixmap(limage))
        self.ProceedVideoLB.setAlignment(QtCore.Qt.AlignCenter)
        self.ProceedVideoLB.setScaledContents(True)
        self.ProceedVideoLB.setMinimumSize(1, 1)

        self.releaseButton()
        self.statusbar.setText("Ready to Compare Between videos")

    def exitButtonPressed(self):
        quit()

    def startButtonPressed(self):

        if self.isthreadActive:
            return

        self.startButton.setEnabled(False)
        self.compareButton.setEnabled(False)

        self.timer = perpetualTimer(1.0 / self.fps, self.updateFrame)
        self.timer.start()

        self.pauseButton.setEnabled(True)
        self.isthreadActive = True

    def slider_value_changed(self):
        if not self.isthreadActive:
            # print('slidervalue change')
            self.horizontalSliderIncrease(0)

    def horizontalSliderIncrease(self, val):
        if self.sliderbusy:
            return
        print("Increase")
        self.sliderbusy = True
        # print(self.horizontalSlider.value())
        # self.horizontalSlider.setValue(self.horizontalSlider.value()+val)
        # print(self.frameID)
        self.frameID = self.stframe + self.horizontalSlider.value() - 1
        # print(self.frameID)
        # self.drawmin=1
        if self.startButton.isEnabled():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frameID)
            ret, frame = self.cap.read()
            self.limg = frame
            # self.on_zoomfit_clicked()
            nchannel = frame.shape[2]
            limg2 = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            timg = cv2.resize(limg2, (int(limg2.shape[1]), int(limg2.shape[0])))
            limage = QtGui.QImage(timg.data, timg.shape[1], timg.shape[0], nchannel * timg.shape[1],
                                  QtGui.QImage.Format_RGB888)

            self.OriginVideoLB.setPixmap(QtGui.QPixmap(limage))
            # self.OriginVideoLB.setAlignment(QtCore.Qt.AlignCenter)
            # self.OriginVideoLB.setScaledContents(True)
            # self.OriginVideoLB.setMinimumSize(1, 1)

        if self.isProceed and self.compareButton.isEnabled():
            self.procCap.set(cv2.CAP_PROP_POS_FRAMES, self.frameID)
            ret, image = self.procCap.read()
            self.limg = image
            nchannel = image.shape[2]

            limg2 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            timg = cv2.resize(limg2, (int(limg2.shape[1]), int(limg2.shape[0])))
            limage = QtGui.QImage(timg.data, timg.shape[1], timg.shape[0], nchannel * timg.shape[1],
                                  QtGui.QImage.Format_RGB888)

            self.ProceedVideoLB.setPixmap(QtGui.QPixmap(limage))
            self.ProceedVideoLB.setAlignment(QtCore.Qt.AlignCenter)
            self.ProceedVideoLB.setScaledContents(True)
            self.ProceedVideoLB.setMinimumSize(1, 1)

        self.sliderbusy = False

    def updateFrame(self):
        self.frameID += 1

    def horizontalSliderSet(self, cnt):
        if cnt + 1 - self.stframe > self.horizontalSlider.maximum() or self.sliderbusy:
            return
        self.sliderbusy2 = True
        self.horizontalSlider.setValue(cnt + 1 - self.stframe)
        tsec = cnt / self.fps
        tmin = int(tsec / 60)
        ttsec = int(tsec - 60 * tmin)
        ksec = tsec - 60 * tmin - ttsec

        self.statusbar.setText(
            "Frame Time: " + str(tmin).zfill(2) + ":" + str(ttsec).zfill(2) + ":" + str(int(ksec * 100)))
        self.sliderbusy2 = False

    def horizontalSliderPressed(self):
        self.sliderbusy = True

    def horizontalSliderReleased(self):
        self.frameID = self.stframe + self.horizontalSlider.value() - 1

        self.drawmin = 1

        if self.startButton.isEnabled():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frameID)
            ret, frame = self.cap.read()
            self.limg = frame
            nchannel = frame.shape[2]
            limg2 = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            timg = cv2.resize(limg2, (int(limg2.shape[1]), int(limg2.shape[0])))
            limage = QtGui.QImage(timg.data, timg.shape[1], timg.shape[0], nchannel * timg.shape[1],
                                  QtGui.QImage.Format_RGB888)
            self.OriginVideoLB.setPixmap(QtGui.QPixmap(limage))
            # self.OriginVideoLB.setAlignment(QtCore.Qt.AlignCenter)
            # self.OriginVideoLB.setScaledContents(True)
            # self.OriginVideoLB.setMinimumSize(1, 1)

        if self.isProceed and self.compareButton.isEnabled():
            self.procCap.set(cv2.CAP_PROP_POS_FRAMES, self.frameID)
            ret, image = self.procCap.read()
            self.limg = image
            nchannel = image.shape[2]

            limg2 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            timg = cv2.resize(limg2, (int(limg2.shape[1]), int(limg2.shape[0])))
            limage = QtGui.QImage(timg.data, timg.shape[1], timg.shape[0], nchannel * timg.shape[1],
                                  QtGui.QImage.Format_RGB888)

            self.ProceedVideoLB.setPixmap(QtGui.QPixmap(limage))
            self.ProceedVideoLB.setAlignment(QtCore.Qt.AlignCenter)
            self.ProceedVideoLB.setScaledContents(True)
            self.ProceedVideoLB.setMinimumSize(1, 1)

        self.sliderbusy = False

    def pauseButtonPressed(self):

        if not self.isthreadActive:
            return

        if self.isProceed:
            self.startButton.setEnabled(False)
            self.compareButton.setEnabled(True)
        else:
            self.startButton.setEnabled(True)
        self.pauseButton.setEnabled(False)

        if not self.timer is None:
            self.timer.cancel()

        self.isthreadActive = False


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = main()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
    klistener.join()
