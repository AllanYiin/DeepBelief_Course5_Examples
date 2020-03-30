"""
This code uses the onnx model to detect faces from live video or cameras.
"""
import time

import cv2
import numpy as np
import onnx
import os
os.environ['TRIDENT_BACKEND'] = 'pytorch'
import trident
from trident import *

# onnx runtime
import onnxruntime as ort
import uuid

dataset=load_examples_data('autodrive')
palette=dataset.traindata.palette
num_classes=len(palette)
if isinstance(palette,list) and len(palette[0])==3:
    pass
elif isinstance(palette,OrderedDict) and len(palette.value_list[0])==3:
    palette=palette.value_list


onnx_path = "Models/tiramisu_softmax.onnx"

#載入onnx模型進行檢查，不需要每次都做，只要轉檔後做一次即可
predictor = onnx.load(onnx_path)
onnx.checker.check_model(predictor)
onnx.helper.printable_graph(predictor.graph)
#predictor = backend.prepare(predictor, device="CPU")  # default CPU

#載入onnx模型至onnxruntime
ort_session = ort.InferenceSession(onnx_path)
#取出輸入張量形狀
input_name = ort_session.get_inputs()[0].name

#獲取影像來源
#如果想要改成webcam 改成  cv2.VideoCapture(0)即可
cap = cv2.VideoCapture('drive1.mp4')  # capture from camera

#寫入成VIDEO  (如果沒有要錄製的話可以註解掉 設定為每秒20禎  720*720)
vw = cv2.VideoWriter('autodrive_v6.avi', cv2.VideoWriter_fourcc('M','J','P','G'),30, (720, 720))


#事先初始化需要的函數
norm=normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
resize=resize((224,224),keep_aspect=True,align_corner=True,order=1)
#rescale=rescale(1280/224.0,order=1)

sum = 0
while True:
    #開始讀取影像
    ret, bgr_image = cap.read()
    if bgr_image is None:
        print("no img")
        #關閉VIDEO寫入
        vw.release()
        break

    #這是裁切掉drive1.mp4行車紀錄器畫面中下方的車前蓋
    bgr_image = bgr_image[:512, 128:-128, :]

    #BGR轉RGB
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    #縮放至224
    rgb_image = resize(rgb_image)
    #normalize圖片
    rgb_image=norm(rgb_image.astype(np.float32)/255.0)
    #加入批次維   image_backend_adaptive轉成通道在前pytorch模式
    rgb_image=np.expand_dims(image_backend_adaptive(rgb_image),0).astype(np.float32)

    #開始記錄時間
    time_time = time.time()
    #執行推論
    pred_mask= ort_session.run(None, {input_name: rgb_image})[0][0]
    print("frame:{0}".format(sum))

   #把不要部分切掉
    pred_mask=pred_mask[:, :112, :].transpose([1,2,0])
    #將(224,112)線性縮放至(720,360)
    pred_mask = cv2.resize(pred_mask, (720,360), interpolation=cv2.INTER_LINEAR)

    #argmax取出最大索引，畫出色彩標籤圖
    pred_mask=label2color(np.argmax(pred_mask,-1),palette).astype(np.float32)

    #把影像縮放至(720,360)
    bgr_image = cv2.resize(bgr_image,(720,360),cv2.INTER_AREA)
    #把原圖與色彩標籤圖疊合
    merge_image=np.concatenate([bgr_image,pred_mask],axis=0).astype(np.uint8)
    #完成處理，結束計時
    print(" infer time:{}".format(time.time() - time_time))
    #寫入禎
    vw.write(merge_image)
    sum+=1
    #顯示禎
    cv2.imshow('annotated', merge_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#關閉影片數據
cap.release()
#銷毀所有視窗
cv2.destroyAllWindows()

