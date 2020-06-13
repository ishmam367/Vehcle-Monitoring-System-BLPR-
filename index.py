import requests
import pyautogui
import uuid
import threading


# This function is responsible for saving
def SaveFileToServer(fileName):
    url = "http://localhost:5000/upload-file"
    files = {'file': open(fileName,'rb')}

    r=requests.post(url, files=files)



def autCaptureImages():
    threading.Timer(5.0, autCaptureImages).start() # this function will trigger after .4 second 
    myScreenshot = pyautogui.screenshot()
    filename=str(uuid.uuid4())
    myScreenshot.save(filename+"razu.png")
    SaveFileToServer(filename+"razu.png")
    print("HTTP Request sent.")

autCaptureImages()




