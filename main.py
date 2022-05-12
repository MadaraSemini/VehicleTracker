import cv2
import numpy as np

cap=cv2.VideoCapture('Video2.mp4')
targetWidthAndHeight=320
confThreshold=0.5
nmsThreshold=0.3

classFile='names.txt'
classes=[]
with open(classFile,'rt') as file:
    classes=file.read().strip('\n').split('\n')
# print(classes)
# print(len(classes))

modelWeights='yolov3-tiny.weights'
modelConfig='yolov3-tiny.cfg'

network=cv2.dnn.readNetFromDarknet(modelConfig,modelWeights)
network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def detectObjects(outputs,img):
    hT,wT,cT=img.shape
    bbox=[]
    classIds=[]
    confs=[]

    for output in outputs:
        for det in output:
            scores=det[5:]
            classId=np.argmax(scores)
            confidence=scores[classId]
            if confidence>confThreshold:
                w,h=int(det[2]*wT),int(det[3]*hT)
                x,y=int((det[0]*wT)-w/2),int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))

    # print(len(bbox))
    indices=cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)
    #print(indices)
    for i in indices:

        box=bbox[i]
        x,y,w,h=box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.putText(img,f'{classes[classIds[i]].upper()}{int(confs[i]*100)}%',
                    (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)



while True:

    try:
        success,img=cap.read()
    except KeyboardInterrupt:
        continue

    blob=cv2.dnn.blobFromImage(img,1/255,(targetWidthAndHeight,targetWidthAndHeight),[0,0,0],1,crop=False)
    network.setInput(blob)

    layerNames=network.getLayerNames()
    #print(layerNames)
    outputNames=[layerNames[i-1] for i in network.getUnconnectedOutLayers()]
    # print(outputNames)
    # print(network.getUnconnectedOutLayers())
    try:
        outputs=network.forward(outputNames)
    except KeyboardInterrupt:
        continue
    # print(outputs[0].shape)
    # print(outputs[1].shape)
    # print(outputs[2].shape)
    # print(outputs[0][0])

    detectObjects(outputs,img)

    cv2.imshow("Image",img)
    cv2.waitKey(1)



