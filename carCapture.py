import cv2
import time
#print(cv2.__version__)



cascade_src = 'cars.xml'
video_src = 'v3.avi'


start_time = time.time()
cap = cv2.VideoCapture(video_src)
#framerate = cap.get(5)
car_cascade = cv2.CascadeClassifier(cascade_src)

i=0
while True:
    i=i+1
    ret, img = cap.read()
    if (type(img) == type(None)):
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    for (x,y,w,h) in cars:
        box=cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        
    cv2.imshow('video', img)    

    cv2.imwrite("template {0}.jpg".format(i),box)
    print("--- %s seconds ---" % (time.time() - start_time))
    time.sleep(2)

    
    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()
