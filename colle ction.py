import cv2
import numpy as np
from keras.models import load_model
import webbrowser
model=load_model('model_file.h5')
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)
while True:
    
    ret,frm=cap.read()#reading frames from webcam
    imgcount=0
    cv2.imshow("window",frm)#showing frame on video
    if cv2.waitKey(1)==27:
        
        cv2.destroyAllWindows()
        cap.release()
        break
    elif (cv2.waitKey(1))%256==32:
        imgcap="opencv_frame_{}.png".format(imgcount)
        cv2.imwrite(imgcap,frm)
        imgcount=imgcount+1
labelsdict={0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'Sad',6:'Suprise'}
frame=cv2.imread("opencv_frame_0.png")
gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
faces=faceDetect.detectMultiScale(gray,1.3,3)
for x,y,w,z in faces:
    faceimg=gray[y:y+z,x:x+w]
    resized=cv2.resize(faceimg,(48,48))
    normalise=resized/255.0
    reshaped=np.reshape(normalise,(1,48,48,1))
    result=model.predict(reshaped)
    label=np.argmax(result,axis=1)[0]
    emotion=labelsdict[label]
    print(emotion)
webbrowser.open(f"https://www.youtube.com/results?search_query={emotion}+songs")

