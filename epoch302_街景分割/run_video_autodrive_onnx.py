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

def label2color(label_mask,palette):
    color_label = np.zeros((*label_mask.shape, 3)).astype(np.float32)
    nums=np.unique(label_mask, return_inverse=True)
    for i in nums[0]:
        if i>0:
            color_label[label_mask==i]=palette[i]
    return color_label

label_path = "models/voc-model-labels.txt"

onnx_path = "Models/tiramisu.onnx"

predictor = onnx.load(onnx_path)
onnx.checker.check_model(predictor)
onnx.helper.printable_graph(predictor.graph)
#predictor = backend.prepare(predictor, device="CPU")  # default CPU

ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name



cap = cv2.VideoCapture('drive1.mp4')  # capture from camera

vw = cv2.VideoWriter('autodrive_v3.avi', cv2.VideoWriter_fourcc('M','J','P','G'),30, (720, 720))
import scipy.ndimage as ndimage
threshold = 0.8
norm=normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
resize=resize((224,224),keep_aspect=True,align_corner=True)
#rescale=rescale(1280/224.0,order=1)

#background = np.ones((480,640,3)) * (255,128,255)
lower_black = np.array([200,200,200], dtype = "uint16")
upper_black = np.array([255,255,255], dtype = "uint16")



sum = 0
while True:
    ret, bgr_image = cap.read()
    #array2image(bgr_image).save('test0.jpg')
    if bgr_image is None:
        print("no img")
        vw.release()
        break


    bgr_image = bgr_image[:512, 128:-128, :]



    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    #array2image(rgb_image).save('test1.jpg')
    rgb_image = resize(rgb_image)
    #rgb_image=clahe()(rgb_image)
   #array2image(rgb_image).save('test2.jpg')
    rgb_image=norm(rgb_image.astype(np.float32)/255.0)

    rgb_image=np.expand_dims(image_backend_adaptive(rgb_image),0).astype(np.float32)

    # confidences, boxes = predictor.run(image)
    time_time = time.time()
    pred_mask= ort_session.run(None, {input_name: rgb_image})[0][0]
    print("frame:{0}".format(sum))

    #time_time = time.time()
    pred_mask=pred_mask[:, :112, :].transpose([1,2,0])
    pred_mask = cv2.resize(pred_mask, (720,360), interpolation=cv2.INTER_LINEAR)

    pred_mask=label2color(np.argmax(pred_mask,-1),palette)
    #pred_mask=cv2.resize(pred_mask.astype(np.uint8),(720,360),cv2.INTER_AREA)
    #print(" decode color time:{}".format(time.time() - time_time))

    #time_time = time.time()

    bgr_image = cv2.resize(bgr_image,(720,360),cv2.INTER_AREA)

    merge_image=np.concatenate([bgr_image,pred_mask],axis=0).astype(np.uint8)
    #print(" merge image time:{}".format(time.time() - time_time))
    print(" infer time:{}".format(time.time() - time_time))
    vw.write(merge_image)
    sum+=1
    cv2.imshow('annotated', merge_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

