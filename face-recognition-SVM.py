import numpy as np

import pandas as pd

import cv2

from sklearn.svm import SVC

face1=np.load('gop.npy').reshape(100,50*50*3)

face2=np.load('shankar.npy').reshape(100,50*50*3)

data=np.concatenate([face1,face2])

dataset=cv2.CascadeClassifier('hr.xml')

labels=np.zeros((200,1))

labels[:100,:]=0.0

labels[100:,:]=1.0

user_name={0:"gopal",1:"shankar"}

font=cv2.FONT_HERSHEY_COMPLEX

capture=cv2.VideoCapture(0)

cvf=SVC()

while True:
    
    ret,img=capture.read()
    
    if ret:
        
        grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        faces=dataset.detectMultiScale(grey)
        
        for x,y,w,h in faces:
            
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            
            myface=img[y:y+h,x:x+w,:]
            
            cvf.fit(data,labels)
            
            myface=cv2.resize(myface,(50,50))
            
            myface=myface.reshape(1,50*50*3)
            
            user=cvf.predict(myface)
            
            cv2.putText(img,user_name[int(user)],(x,y),font,1,(0,255,0),2)

        cv2.imshow('result',img)
        
        if cv2.waitKey(1)==27:

            break
            
    else:

        print("camera not working")

capture.release()

cv2.destroyAllWindows()
