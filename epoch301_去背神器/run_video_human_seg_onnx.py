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
from caffe2.python.onnx import backend

# onnx runtime
import onnxruntime as ort
import uuid



label_path = "models/voc-model-labels.txt"

onnx_path = "Models/deeplab_v3plus_matting.onnx"

predictor = onnx.load(onnx_path)
onnx.checker.check_model(predictor)
onnx.helper.printable_graph(predictor.graph)
#predictor = backend.prepare(predictor, device="CPU")  # default CPU

ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name




cap = cv2.VideoCapture(0)  # capture from camera
#out = cv2.VideoWriter('C:/Users/Allan/Downloads/ATM4_revised.avi', cv2.VideoWriter_fourcc(*'DIVX'), 4, (1920, 1080))
import scipy.ndimage as ndimage
threshold = 0.8
norm=normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
resize=resize((224,224),keep_aspect=True,align_corner=True)
rescale=rescale(640/224.0,order=0)
background =cv2.imread('cosmo.jpg')
#background = np.ones((480,640,3)) * (255,128,255)
lower_black = np.array([200,200,200], dtype = "uint16")
upper_black = np.array([255,255,255], dtype = "uint16")



sum = 0
while True:
    ret, bgr_image = cap.read()
    #array2image(bgr_image).save('test0.jpg')
    if bgr_image is None:
        print("no img")
        break
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

    #trimap=mask2trimap((3,5))(np.argmax(pred_mask, 0))
    pred_mask=pred_mask[1,:,:]#*np.argmax(pred_mask,0)#*(trimap!=0.5)+pred_mask[1, :, :] *(trimap==0.5)
    pred_mask[pred_mask>0.9]=1
    pred_mask[pred_mask < 0.2] = 0
    print("infer time:{}".format(time.time() - time_time))


    # pred_mask[pred_mask>0.8]=1
    # pred_mask[pred_mask <0.6] = 0


    pred_mask=rescale(pred_mask)
    # black_mask = cv2.inRange(bgr_image, lower_black, upper_black)
    # pred_mask[black_mask]=0
    pred_mask=np.expand_dims(pred_mask[:480,:640],-1)


    #array2mask(pred_mask[:,:,0]*255).save('test3.jpg')
    bgr_image=pred_mask * bgr_image + (1 - pred_mask) * background

    bgr_image = cv2.resize(bgr_image,(1280,960),interpolation=cv2.INTER_AREA)
    # bgr_image[:60,:,:]=0
    # bgr_image[-60:, :, :] = 0
    #array2image(bgr_image).save('test4.jpg')
    cv2.imshow('annotated', bgr_image.astype(np.uint8))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
print("sum:{}".format(sum))
