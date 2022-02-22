import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(0)
while True:
    _, image = camera.read()
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayscale, 1.1, 4)
    for(x,y,w,h) in faces:
        cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)
    cv2.imshow('image', image)

    k = cv2.waitKey(30) & 0xff
    if (k == 27):
        break

camera.release()