# -*- coding: utf-8 -*-

import dlib  # 人脸识别的库dlib
import cv2  # 图像处理的库OpenCv
import wx  # 构造显示界面的GUI
import wx.xrc
import wx.adv
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np  # 数据处理的库 numpy
import time
import math
import pyttsx3
import pythoncom
from win32com import client
import sats2

COVER = 'E:/Fatigue_detect/images/ui8.png'

class Fatigue_detecting(wx.Frame):

    def __init__(self, parent, title):
        wx.Frame.__init__(self, parent, id=wx.ID_ANY, title=title, pos=wx.DefaultPosition, size=wx.Size(925, 535),
                          style=wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL)

        self.SetSizeHints(wx.DefaultSize, wx.DefaultSize)
        self.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_MENU))

        bSizer1 = wx.BoxSizer(wx.VERTICAL)
        bSizer2 = wx.BoxSizer(wx.HORIZONTAL)
        bSizer3 = wx.BoxSizer(wx.VERTICAL)

        self.m_animCtrl1 = wx.adv.AnimationCtrl(self, wx.ID_ANY, wx.adv.NullAnimation, wx.DefaultPosition,
                                                wx.DefaultSize, wx.adv.AC_DEFAULT_STYLE)
        bSizer3.Add(self.m_animCtrl1, 1, wx.ALL | wx.EXPAND, 5)
        bSizer2.Add(bSizer3, 9, wx.EXPAND, 5)
        bSizer4 = wx.BoxSizer(wx.VERTICAL)
        sbSizer1 = wx.StaticBoxSizer(wx.StaticBox(self, wx.ID_ANY, u"参数设置"), wx.VERTICAL)
        sbSizer2 = wx.StaticBoxSizer(wx.StaticBox(sbSizer1.GetStaticBox(), wx.ID_ANY, u"基础操作"), wx.VERTICAL)
        gSizer1 = wx.GridSizer(0, 2, 0, 8)
        self.m_choice1 = wx.Button(sbSizer2.GetStaticBox(), wx.ID_ANY, u"加载车载摄像头", wx.DefaultPosition, wx.Size(120, 60),
                                   0)
        gSizer1.Add(self.m_choice1, 0, wx.ALL, 5)
        self.camera_button1 = wx.Button(sbSizer2.GetStaticBox(), wx.ID_ANY, u"开始检测", wx.DefaultPosition,
                                        wx.Size(120, 60), 0)
        gSizer1.Add(self.camera_button1, 0, wx.ALL, 5)
        self.off_button3 = wx.Button(sbSizer2.GetStaticBox(), wx.ID_ANY, u"暂停", wx.DefaultPosition, wx.Size(120, 60), 0)
        gSizer1.Add(self.off_button3, 0, wx.ALL, 5)
        self.off_button4 = wx.Button(sbSizer2.GetStaticBox(), wx.ID_ANY, u"退出检测", wx.DefaultPosition, wx.Size(120, 60),
                                     0)
        gSizer1.Add(self.off_button4, 0, wx.ALL, 5)
        sbSizer2.Add(gSizer1, 1, wx.EXPAND, 5)
        sbSizer1.Add(sbSizer2, 2, wx.EXPAND, 5)
        sbSizer6 = wx.StaticBoxSizer(wx.StaticBox(sbSizer1.GetStaticBox(), wx.ID_ANY, u"状态输出"), wx.VERTICAL)
        self.m_textCtrl3 = wx.TextCtrl(sbSizer6.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition,
                                       wx.DefaultSize, wx.TE_MULTILINE | wx.TE_READONLY)
        sbSizer6.Add(self.m_textCtrl3, 1, wx.ALL | wx.EXPAND, 5)
        sbSizer1.Add(sbSizer6, 5, wx.EXPAND, 5)
        bSizer4.Add(sbSizer1, 1, wx.EXPAND, 5)
        bSizer2.Add(bSizer4, 3, wx.EXPAND, 5)
        bSizer1.Add(bSizer2, 1, wx.EXPAND, 5)

        self.SetSizer(bSizer1)
        self.Layout()
        self.Centre(wx.BOTH)

        # Connect Events
        self.m_choice1.Bind(wx.EVT_BUTTON, self.prepare)
        self.camera_button1.Bind(wx.EVT_BUTTON, self.camera_on)  # 开
        self.off_button3.Bind(wx.EVT_BUTTON, self.off)  # 关
        self.off_button4.Bind(wx.EVT_BUTTON, self.exit)
        # 封面图片
        self.image_cover = wx.Image(COVER, wx.BITMAP_TYPE_ANY)
        # 显示图片在m_animCtrl1上
        self.bmp = wx.StaticBitmap(self.m_animCtrl1, -1, wx.Bitmap(self.image_cover))

        # 设置窗口标题的图标
        self.icon = wx.Icon('E:/Fatigue_detect/images/555.ico', wx.BITMAP_TYPE_ICO)
        self.SetIcon(self.icon)
        # 系统事件
        self.Bind(wx.EVT_CLOSE, self.OnClose)


        """参数"""
        # 默认为摄像头0
        self.VIDEO_STREAM = 0
        self.CAMERA_STYLE = False  # False未打开摄像头，True摄像头已打开
        self.AR_CONSEC_FRAMES_check = 3
        self.OUT_AR_CONSEC_FRAMES_check = 5
        # 闪烁阈值（秒）
        # 眼睛长宽比
        self.EYE_AR_THRESH = 0.2
        self.EYE_AR_CONSEC_FRAMES = 3
        # 打哈欠长宽比
        self.MAR_THRESH = 0.5
        self.MOUTH_AR_CONSEC_FRAMES = 3
        # 瞌睡点头
        self.HAR_THRESH = 0.3
        self.NOD_AR_CONSEC_FRAMES = 5

        """计数"""
        # 初始化帧计数器和眨眼总数
        self.COUNTER = 0
        self.TOTAL = 0
        # 初始化帧计数器和打哈欠总数
        self.mCOUNTER = 0
        self.mTOTAL = 0
        # 初始化帧计数器和点头总数
        self.hCOUNTER = 0
        self.hTOTAL = 0
        # 离职时间长度
        self.oCOUNTER = 0
        # 初始化眨眼频率,点头频率，打哈欠频率
        self.frequency = 0
        self.hfrequency = 0
        self.yfrequency = 0
        # 初始化疲劳程度
        self.score = 0

        """姿态"""
        # 世界坐标系(UVW)：填写3D参考点，该模型参考http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp
        self.object_pts = np.float32([[6.825897, 6.760612, 4.402142],  # 33左眉左上角
                                      [1.330353, 7.122144, 6.903745],  # 29左眉右角
                                      [-1.330353, 7.122144, 6.903745],  # 34右眉左角
                                      [-6.825897, 6.760612, 4.402142],  # 38右眉右上角
                                      [5.311432, 5.485328, 3.987654],  # 13左眼左上角
                                      [1.789930, 5.393625, 4.413414],  # 17左眼右上角
                                      [-1.789930, 5.393625, 4.413414],  # 25右眼左上角
                                      [-5.311432, 5.485328, 3.987654],  # 21右眼右上角
                                      [2.005628, 1.409845, 6.165652],  # 55鼻子左上角
                                      [-2.005628, 1.409845, 6.165652],  # 49鼻子右上角
                                      [2.774015, -2.080775, 5.048531],  # 43嘴左上角
                                      [-2.774015, -2.080775, 5.048531],  # 39嘴右上角
                                      [0.000000, -3.116408, 6.097667],  # 45嘴中央下角
                                      [0.000000, -7.415691, 4.070434]])  # 6下巴角

        # 相机坐标系(XYZ)：添加相机内参
        self.K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
                  0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
                  0.0, 0.0, 1.0]  # 等价于矩阵[fx, 0, cx; 0, fy, cy; 0, 0, 1]
        # 图像中心坐标系(uv)：相机畸变参数[k1, k2, p1, p2, k3]
        self.D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

        # 像素坐标系(xy)：填写凸轮的本征和畸变系数
        self.cam_matrix = np.array(self.K).reshape(3, 3).astype(np.float32)
        self.dist_coeffs = np.array(self.D).reshape(5, 1).astype(np.float32)

        # 重新投影3D点的世界坐标轴以验证结果姿势
        self.reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                                        [10.0, 10.0, -10.0],
                                        [10.0, -10.0, -10.0],
                                        [10.0, -10.0, 10.0],
                                        [-10.0, 10.0, 10.0],
                                        [-10.0, 10.0, -10.0],
                                        [-10.0, -10.0, -10.0],
                                        [-10.0, -10.0, 10.0]])
        # 绘制正方体12轴
        self.line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
                           [4, 5], [5, 6], [6, 7], [7, 4],
                           [0, 4], [1, 5], [2, 6], [3, 7]]

    def __del__(self):
        pass

    def get_head_pose(self, shape):  # 头部姿态估计
        # （像素坐标集合）填写2D参考点，注释遵循https://ibug.doc.ic.ac.uk/resources/300-W/
        # 17左眉左上角/21左眉右角/22右眉左上角/26右眉右上角/36左眼左上角/39左眼右上角/42右眼左上角/
        # 45右眼右上角/31鼻子左上角/35鼻子右上角/48左上角/54嘴右上角/57嘴中央下角/8下巴角
        image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                                shape[39], shape[42], shape[45], shape[31], shape[35],
                                shape[48], shape[54], shape[57], shape[8]])
        # solvePnP计算姿势——求解旋转和平移矩阵：
        # rotation_vec表示旋转矩阵，translation_vec表示平移矩阵，cam_matrix与K矩阵对应，dist_coeffs与D矩阵对应。
        _, rotation_vec, translation_vec = cv2.solvePnP(self.object_pts, image_pts, self.cam_matrix, self.dist_coeffs)
        # projectPoints重新投影误差：原2d点和重投影2d点的距离（输入3d点、相机内参、相机畸变、r、t，输出重投影2d点）
        reprojectdst, _ = cv2.projectPoints(self.reprojectsrc, rotation_vec, translation_vec, self.cam_matrix,
                                            self.dist_coeffs)
        reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))  # 以8行2列显示

        # 计算欧拉角calc euler angle
        # 参考https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#decomposeprojectionmatrix
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)  # 罗德里格斯公式（将旋转矩阵转换为旋转向量）
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))  # 水平拼接，vconcat垂直拼接
        # decomposeProjectionMatrix将投影矩阵分解为旋转矩阵和相机矩阵
        _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

        pitch, yaw, roll = [math.radians(_) for _ in euler_angle]

        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))
        # print('pitch:{}, yaw:{}, roll:{}'.format(pitch, yaw, roll))

        return reprojectdst, euler_angle  # 投影误差，欧拉角

    def eye_aspect_ratio(self, eye):
        # 垂直眼标志（X，Y）坐标
        A = dist.euclidean(eye[1], eye[5])  # 计算两个集合之间的欧式距离
        B = dist.euclidean(eye[2], eye[4])
        # 计算水平之间的欧几里得距离
        # 水平眼标志（X，Y）坐标
        C = dist.euclidean(eye[0], eye[3])
        # 眼睛长宽比的计算
        ear = (A + B) / (2.0 * C)
        # 返回眼睛的长宽比
        return ear

    def mouth_aspect_ratio(self, mouth):  # 嘴部
        A = np.linalg.norm(mouth[2] - mouth[9])  # 51, 59
        B = np.linalg.norm(mouth[4] - mouth[7])  # 53, 57
        C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
        mar = (A + B) / (2.0 * C)
        return mar

    def _learning_face(self, event):
        """dlib的初始化调用"""
        # 使用人脸检测器get_frontal_face_detector
        self.detector = dlib.get_frontal_face_detector()
        # dlib的68点模型，使用作者训练好的特征预测器
        self.predictor = dlib.shape_predictor("E:/Fatigue_detect/model/shape_predictor_68_face_landmarks.dat")
        self.m_textCtrl3.AppendText(u"加载模型成功!!!\n")
        # 分别获取左右眼面部标志的索引

        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

        # 建cv2摄像头对象，这里使用电脑自带摄像头，如果接了外部摄像头，则自动切换到外部摄像头
        self.cap = cv2.VideoCapture(self.VIDEO_STREAM)

        if self.cap.isOpened() == True:  # 返回true/false 检查初始化是否成功
            self.CAMERA_STYLE = True
            self.m_textCtrl3.AppendText(u"打开摄像头成功!!!\n")
            pythoncom.CoInitialize()
            engine = client.Dispatch("SAPI.SpVoice")
            engine.Speak('打开摄像头成功，开始为您检测，祝您一路顺风')
        else:
            engine = pyttsx3.init()
            content = "打开摄像头失败，请重试"
            print('准备开始语音播报...')
            # 设置要播报的Unicode字符串
            engine.say(content)
            # 等待语音播报完毕
            engine.runAndWait()
            self.m_textCtrl3.AppendText(u"摄像头打开失败!!!\n")
            # 显示封面图
            self.bmp.SetBitmap(wx.Bitmap(self.image_cover))
        # 成功打开视频，循环读取视频流
        while (self.cap.isOpened()):
            # cap.read()
            # 返回两个值：
            #    一个布尔值true/false，用来判断读取视频是否成功/是否到视频末尾
            #    图像对象，图像的三维矩阵
            flag, im_rd = self.cap.read()
            start = time.time()
            # 取灰度
            img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)

            # 使用人脸检测器检测每一帧图像中的人脸。并返回人脸数faces
            faces = self.detector(img_gray, 0)
            # 如果检测到人脸
            if (len(faces) != 0):
                # enumerate方法同时返回数据对象的索引和数据，k为索引，d为faces中的对象
                for k, d in enumerate(faces):
                    # 用红色矩形框出人脸
                    cv2.rectangle(im_rd, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255), 1)
                    # 使用预测器得到68点数据的坐标
                    shape = self.predictor(im_rd, d)
                    # 圆圈显示每个特征点
                    for i in range(68):
                        cv2.circle(im_rd, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), -1, 8)
                    # 将脸部特征信息转换为数组array的格式
                    shape = face_utils.shape_to_np(shape)
                    """
                    打哈欠
                    """
                    # if self.yawn_checkBox1.GetValue()== True:
                    # 嘴巴坐标
                    mouth = shape[mStart:mEnd]
                    # 打哈欠
                    mar = self.mouth_aspect_ratio(mouth)
                    # 使用cv2.convexHull获得凸包位置，使用drawContours画出轮廓位置进行画图操作
                    mouthHull = cv2.convexHull(mouth)
                    cv2.drawContours(im_rd, [mouthHull], -1, (0, 255, 0), 1)
                    # 同理，判断是否打哈欠
                    if mar > self.MAR_THRESH:  # 张嘴阈值0.5
                        self.mCOUNTER += 1
                    else:
                        # 如果连续3次都小于阈值，则表示打了一次哈欠
                        if self.mCOUNTER >= self.MOUTH_AR_CONSEC_FRAMES:  # 阈值：3
                            self.mTOTAL += 1
                            # 显示
                            # cv2.putText(im_rd, "Yawning!", (10, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            self.m_textCtrl3.AppendText(time.strftime('%Y-%m-%d %H:%M ', time.localtime()) + u"打哈欠\n")
                        # 重置嘴帧计数器
                        self.mCOUNTER = 0
                    # else:
                    #     pass
                    """
                    眨眼
                    """
                    # if self.blink_checkBox2.GetValue()== True:
                    # 提取左眼和右眼坐标
                    leftEye = shape[lStart:lEnd]
                    rightEye = shape[rStart:rEnd]
                    # 构造函数计算左右眼的EAR值，使用平均值作为最终的EAR
                    leftEAR = self.eye_aspect_ratio(leftEye)
                    rightEAR = self.eye_aspect_ratio(rightEye)
                    ear = (leftEAR + rightEAR) / 2.0
                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    # 使用cv2.convexHull获得凸包位置，使用drawContours画出轮廓位置进行画图操作
                    cv2.drawContours(im_rd, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(im_rd, [rightEyeHull], -1, (0, 255, 0), 1)
                    # 循环，满足条件的，眨眼次数+1
                    if ear < self.EYE_AR_THRESH:  # 眼睛长宽比：0.2
                        self.COUNTER += 1

                    else:
                        # 如果连续3次都小于阈值，则表示进行了一次眨眼活动
                        if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:  # 阈值：3
                            self.TOTAL += 1
                            self.m_textCtrl3.AppendText(time.strftime('%Y-%m-%d %H:%M ', time.localtime()) + u"眨眼\n")
                        # 重置眼帧计数器
                        self.COUNTER = 0
                    # else:
                    #     pass
                    """
                    瞌睡点头
                    """
                    # if self.nod_checkBox7.GetValue()== True:
                    # 获取头部姿态
                    reprojectdst, euler_angle = self.get_head_pose(shape)
                    har = euler_angle[0, 0]  # 取pitch旋转角度
                    if har > self.HAR_THRESH:  # 点头阈值0.3
                        self.hCOUNTER += 1
                    else:
                        # 如果连续3次都小于阈值，则表示瞌睡点头一次
                        if self.hCOUNTER >= self.NOD_AR_CONSEC_FRAMES:  # 阈值：3
                            self.hTOTAL += 1
                            self.m_textCtrl3.AppendText(time.strftime('%Y-%m-%d %H:%M ', time.localtime()) + u"瞌睡点头\n")
                        # 重置点头帧计数器
                        self.hCOUNTER = 0
            else:
                # 没有检测到人脸
                self.oCOUNTER += 1
                cv2.putText(im_rd, "No Face", (350, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3, cv2.LINE_AA)
                if self.oCOUNTER >= self.OUT_AR_CONSEC_FRAMES_check:
                    self.m_textCtrl3.AppendText(time.strftime('%Y-%m-%d %H:%M ', time.localtime()) + u"无人驾驶状态!!!\n")
                    self.oCOUNTER = 0

            # 判断疲劳程度
            if self.score >= 30 and self.score <= 55:
                cv2.putText(im_rd, "mid fatigue", (350, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            if self.score > 55 and self.score <= 75:
                cv2.putText(im_rd, "moderate fatigue", (350, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            if self.score > 75:
                cv2.putText(im_rd, "severe fatigue", (350, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            T = time.time() - start
            fps = 1 / T  # 实时在视频上显示fps
            fps_txt = 'fps:%.2f' % (fps)
            cv2.putText(im_rd, "Blinks: {}".format(self.TOTAL), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(im_rd, "Nod: {}".format(self.hTOTAL), (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(im_rd, "Yawning: {}".format(self.mTOTAL), (210, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(im_rd, "Blink Frequency:{}".format(self.frequency), (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(im_rd, fps_txt, (450, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(im_rd, "fatigue degree:{}".format(self.score), (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(im_rd, "Mar Frequency:{}".format(self.yfrequency), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(im_rd, "Nod Frequency:{}".format(self.hfrequency), (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            # opencv中imread的图片内部是BGR排序，wxPython的StaticBitmap需要的图片是RGB排序，不转换会出现颜色变换
            height, width = im_rd.shape[:2]
            image1 = cv2.cvtColor(im_rd, cv2.COLOR_BGR2RGB)
            pic = wx.Bitmap.FromBuffer(width, height, image1)
            # 显示图片在panel上：
            self.bmp.SetBitmap(pic)
        # 释放摄像头
        self.cap.release()

    def count(self, event):
        for i in range(500):
        #while True:
            fTOTAL = self.TOTAL
            fmTOTAL = self.mTOTAL
            fhTOTAL = self.hTOTAL
            time.sleep(5)
            lTOTAL = self.TOTAL
            lmTOTAL = self.mTOTAL
            lhTOTAL = self.hTOTAL
            # 计算眨眼频率
            self.frequency = (lTOTAL - fTOTAL) / 5
            # 计算点头频率
            self.hfrequency = (lhTOTAL - fhTOTAL) / 5
            # 计算打哈欠频率
            self.yfrequency = (lmTOTAL - fmTOTAL) / 5

            if self.score >= 100:
                self.score = 100
            if self.score <= 0:
                self.score = 0
            if self.frequency > 0.47 and self.frequency < 0.61:
                self.score = self.score + 10
            if self.frequency > 0.62 and self.frequency < 0.95:
                self.score = self.score + 15
            if self.frequency > 0.96:
                self.score = self.score + 20
            if self.frequency < 0.47 and self.score >= 0:
                self.score = self.score - 5
            if self.yfrequency >= 0.2 and self.yfrequency <= 0.4:
                self.score = self.score + 10
            if self.yfrequency > 0.4 and self.yfrequency <= 0.6:
                self.score = self.score + 15
            if self.yfrequency > 0.6:
                self.score = self.score + 20
            if self.yfrequency < 0.2 and self.score >= 0:
                self.score = self.score - 10
            if self.hfrequency >= 0.2 and self.hfrequency <= 0.4:
                self.score = self.score + 15
            if self.hfrequency > 0.4 and self.hfrequency <= 0.6:
                self.score = self.score + 20
            if self.hfrequency > 0.6:
                self.score = self.score + 25
            if self.hfrequency < 0.2 and self.score >= 0:
                self.score = self.score - 20
            if self.score >= 100:
                self.score = 100
            if self.score <= 0:
                self.score = 0

    def alarm(self, event):
        #while True:
        for i in range(500):
            #print("开始进入休眠")
            time.sleep(3)
            #print("结束休眠，进入新的一轮循环")
            if self.score >= 30 and self.score <= 55:
                self.m_textCtrl3.AppendText(
                    time.strftime('%Y-%m-%d %H:%M ', time.localtime()) + u"警报警报已进入轻度疲劳，请打起精神！！\n准备开始语音播报\n")
                pythoncom.CoInitialize()
                engine = client.Dispatch("SAPI.SpVoice")
                engine.Speak('警报警报，检测到您已进入轻度疲劳，请注意')
                # 语音播报内容
                # self.m_textCtrl3.AppendText(time.strftime('%Y-%m-%d %H:%M ', time.localtime()) + u"警报警报已进入轻度疲劳，请打起精神！！\n准备开始语音播报\n")
            if self.score > 55 and self.score <= 75:
                # 语音播报内容
                self.m_textCtrl3.AppendText(
                    time.strftime('%Y-%m-%d %H:%M ', time.localtime()) + u"警报警报已进入中度疲劳，请尽快打起精神！！！\n准备开始语音播报\n")
                pythoncom.CoInitialize()
                engine = client.Dispatch("SAPI.SpVoice")
                engine.Speak('警报警报，检测到您已进入中度疲劳，请尽快打起精神，否则即将自动报警')
            if self.score > 75:
                # 语音播报内容
                self.m_textCtrl3.AppendText(
                    time.strftime('%Y-%m-%d %H:%M ', time.localtime()) + u"警报警报已进入重度疲劳，请靠边停车，已为您自动报警\n准备开始语音播报\n")
                pythoncom.CoInitialize()
                engine = client.Dispatch("SAPI.SpVoice")
                engine.Speak('警报警报，检测到您已进入重度疲劳，请靠边停车，已为您自动报警')

    def camera_on(self, event):
        """使用多线程，子线程运行后台的程序，主线程更新前台的UI，这样不会互相影响"""
        import _thread
        # 创建子线程，按钮调用这个方法，
        _thread.start_new_thread(self._learning_face, (event,))
        _thread.start_new_thread(self.count, (event,))
        _thread.start_new_thread(self.alarm, (event,))

    def off(self, event):
        """关闭摄像头，显示封面页"""
        self.cap.release()
        self.bmp.SetBitmap(wx.Bitmap(self.image_cover))

    def OnClose(self, evt):
        """关闭窗口事件函数"""
        dlg = wx.MessageDialog(None, u'确定要关闭本窗口？', u'操作提示', wx.YES_NO | wx.ICON_QUESTION)
        if (dlg.ShowModal() == wx.ID_YES):
            self.Destroy()
            pythoncom.CoInitialize()
            engine = client.Dispatch("SAPI.SpVoice")
            engine.Speak('再见，欢迎您再次使用，祝您一路平安')
            print("检测结束，成功退出程序!!!")
            sats2.page_draggable_layout()

    def exit(self, evt):
        dlg = wx.MessageDialog(None, u'确定要退出检测吗？', u'操作提示', wx.YES_NO | wx.ICON_QUESTION)
        if (dlg.ShowModal() == wx.ID_YES):
            self.Destroy()
            pythoncom.CoInitialize()
            engine = client.Dispatch("SAPI.SpVoice")
            engine.Speak('再见，欢迎您再次使用，祝您一路平安')
            print("检测结束，成功退出程序!!!")
            sats2.page_draggable_layout()

    def prepare(self, evt):
        self.m_textCtrl3.AppendText(u"加载车载摄像头成功!!!\n")
        pythoncom.CoInitialize()
        engine = client.Dispatch("SAPI.SpVoice")
        engine.Speak('初始化车载摄像头成功')

class main_app(wx.App):
    # OnInit 方法在主事件循环开始前被wxPython系统调用，是wxpython独有的
    def OnInit(self):
        self.frame = Fatigue_detecting(parent=None, title="疲劳驾驶检测")
        self.frame.Show(True)
        return True


if __name__ == "__main__":
    app = main_app()
    app.MainLoop()

