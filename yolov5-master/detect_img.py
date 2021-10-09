import cv2
import detect
import time
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import win32gui,win32ui,win32con
# import pyautogu
cap=cv2.VideoCapture(0)
a=detect.detectapi(weights='weights/yolov5s.pt')
# img = cv2.imread("1629725766.jpg")
# def get_windows():
#     # 获取窗口句柄
#     handle = win32gui.FindWindow(None,'新建文本文档.txt - 记事本')
#     # 将窗口放在前台，并激活该窗口（窗口不能最小化）
#     win32gui.SetForegroundWindow(handle)
#     # 获取窗口DC
#     hdDC = win32gui.GetWindowDC(handle)
#     # 根据句柄创建一个DC
#     newhdDC = win32ui.CreateDCFromHandle(hdDC)
#     # 创建一个兼容设备内存的DC
#     saveDC = newhdDC.CreateCompatibleDC()
#     # 创建bitmap保存图片
#     saveBitmap = win32ui.CreateBitmap()
#
#     # 获取窗口的位置信息
#     left, top, right, bottom = win32gui.GetWindowRect(handle)
#     # 窗口长宽
#     width = right - left
#     height = bottom - top
#     # print(left, top, right, bottom)
#    # bitmap初始化
#     # saveBitmap.CreateCompatibleBitmap(newhdDC, width, height)
#     # saveDC.SelectObject(saveBitmap)
#     # saveDC.BitBlt((0, 0), (width, height), newhdDC, (0, 0), win32con.SRCCOPY)
#     # saveBitmap.SaveBitmapFile(saveDC, "截图.png")
#     return left, top, right, bottom
while True:
    start= time.time()
    rec,img = cap.read()
    result,names =a.detect([img])
    # print(names)
    # print(result)
    img=result[0][0] #第一张图片的处理结果图片
    print(result[0][1])
    cla_reult = result[0][1]
    person_pos = []
    for result in (cla_reult):
        if result[0]==0:
            person_pos.append(result[1])
    '''
    for cls,(x1,y1,x2,y2),conf in result[0][1]: #第一张图片的处理结果标签。
        print(cls,x1,y1,x2,y2,conf)
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0))
        cv2.putText(img,names[cls],(x1,y1-20),cv2.FONT_HERSHEY_DUPLEX,1.5,(255,0,0))
    '''
    print(person_pos)
    print(1/(time.time()-start))
    cv2.imshow("vedio",img)
    if cv2.waitKey(1)==ord('q'):
        break
    # plt.figure()
    # plt.imshow(img)
    # plt.show()