import cv2
import numpy as np
import face_recognition

imgmodi = face_recognition.load_image_file('imagebasic/modi.jpg')
imgmodi = cv2.cvtColor(imgmodi, cv2.COLOR_BGR2RGB)
imgtest = face_recognition.load_image_file('imagebasic/rahul.jpg')
imgtest = cv2.cvtColor(imgtest, cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgmodi)[0]
encodemodi = face_recognition.face_encodings(imgmodi)[0]
cv2.rectangle(imgmodi,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

faceloctest = face_recognition.face_locations(imgtest)[0]
encodetest = face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(faceloctest[3],faceloctest[0]),(faceloctest[1],faceloctest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodemodi],encodetest)
facedis = face_recognition.face_distance([encodemodi],encodetest)
print(results,facedis)
cv2.putText(imgtest,f'{results} {round(facedis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
cv2.imshow('modi', imgmodi)
cv2.imshow('testmodi', imgtest)
cv2.waitKey(0)