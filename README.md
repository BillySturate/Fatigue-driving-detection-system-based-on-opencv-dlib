##                       基于opencv,dlib的疲劳驾驶检测系统

### 1.宣传海报

![](Fatigue driving detection system/images/Wake-基于面部多信息特征融合的疲劳驾驶检测系统.jpg)



### 2.效果展示

#### 2.1ui界面设计

![](https://raw.githubusercontent.com/Billysturate/Fatigue-driving-detection-system-based-on-opencv-dlib/main/Fatigue%20driving%20detection%20system/images/123.png)

#### 2.2效果演示

![](https://raw.githubusercontent.com/Billysturate/Fatigue-driving-detection-system-based-on-opencv-dlib/main/Fatigue%20driving%20detection%20system/images/%E6%BC%94%E7%A4%BA%E7%94%BB%E9%9D%A2.png)

#### 2.4报表界面部分演示

![](https://raw.githubusercontent.com/Billysturate/Fatigue-driving-detection-system-based-on-opencv-dlib/main/Fatigue%20driving%20detection%20system/images/%E6%89%87%E5%BD%A2%E5%9B%BE.png)

![](https://raw.githubusercontent.com/Billysturate/Fatigue-driving-detection-system-based-on-opencv-dlib/main/Fatigue%20driving%20detection%20system/images/%E6%8A%98%E7%BA%BF%E5%9B%BE.png)



### 3.项目进展

| ~~**眨眼检测**~~            | **DONE** |
| --------------------------- | -------- |
| ~~**打哈欠检测**~~          | **DONE** |
| ~~**瞌睡点头检测**~~        | **DONE** |
| ~~**实时眨眼频率计算**~~    | **DONE** |
| ~~**实时打哈欠频率**~~      | **DONE** |
| ~~**实时瞌睡点头频率**~~    | **DONE** |
| ~~**疲劳程度的计算**~~      | **DONE** |
| ~~**疲劳程度划分**~~        | **DONE** |
| ~~**UI界面设计**~~          | **DONE** |
| ~~**FPS计算以及语音播报**~~ | **DONE** |
| ~~**报表界面设计**~~        | **DONE** |



### 4.界面参数介绍

| Blinks               | 眨眼次数             |
| -------------------- | -------------------- |
| **Yawning**          | **打哈欠次数**       |
| **Nod**              | **瞌睡点头次数**     |
| **Blink Frequency**  | **实时眨眼频率**     |
| **Yawing Frequency** | **实时打哈欠频率**   |
| **Nod Frequency**    | **实时瞌睡点头频率** |
| **Score**            | **疲劳程度**         |
| **FPS**              | **视频帧率**         |

### 5.文件介绍

main.py-------------------------主程序

sats2.py--------------------------报表界面制作(pyecharts,感觉还是挺好用的，具体操作可以看官网文档，中国人开发的库值得推荐)

[pyecharts gallery](https://gallery.pyecharts.org/#/Bar/stack_bar_percent)								[pyecharts操作手册](https://pyecharts.org/#/zh-cn/intro)

images目录下存放图片png以及ico

model目录下存放68人脸关键点模型

fatigue_detect.html-----------------可视化报表















