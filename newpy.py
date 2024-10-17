import requests
import json
import numpy as np
import face_recognition
import os
from datetime import datetime
import cv2
import tkinter as tk
from PIL import Image, ImageTk

# database config
API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNqandtY2xzcHBwaHZnamhja2dxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MDQ4MjI0MDQsImV4cCI6MjAyMDM5ODQwNH0.vF3qXyZUjXDHS-u1sgcDdC6lq_6LUQlABIE5ekC6xhw'
DATABASE_URL = 'https://cjjwmclsppphvgjhckgq.supabase.co/rest/v1/attendance?select=*'



#initiallize a Tkinter root object
root = tk.Tk()

# from PIL import ImageGrab
path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(img)

        if len(faces) > 0:
            encode = face_recognition.face_encodings(img, faces)[0]
            encodeList.append(encode)

    return encodeList
def is_attendance_taken(name, database_url, api_key):
    now = datetime.now()
    current_date = now.strftime('%Y-%m-%d')

    headers = {
        'apikey': api_key,
        'Authorization': f"Bearer {api_key}",
        'Content-Type': 'application/json'
    }

    response = requests.get(database_url, headers=headers)
    entries = response.json() if response.status_code == 200 else []

    for entry in entries:
        entry_date = entry.get('time', '').split(' ')[0]  # Extracting date part
        entry_name = entry.get('name', '')

        if entry_date == current_date and entry_name == name:
            return True

    return False

def send_attendance_request(name, database_url, api_key):
    now = datetime.now()
    dtString = now.strftime('%Y-%m-%d %H:%M:%S')

    new_entry = [{"name": name, "time": dtString}]
    headers = {
        'apikey': api_key,
        'Authorization': f"Bearer {api_key}",
        'Content-Type': 'application/json'
    }

    response = requests.post(database_url, headers=headers, data=json.dumps(new_entry))
    print(f"New attendance row inserted in Database [+] Status code: {response.status_code}")

def markAttendance(name):
    result = is_attendance_taken(name, DATABASE_URL, API_KEY)
    if result:
        print(f"Attendance found for {name} today: {result}")
    else:
        send_attendance_request(name, DATABASE_URL, API_KEY)


#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr

encodeListKnown = findEncodings(images)
print('Encoding Complete')


def show():
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                markAttendance(name)

        cv2.imshow('Webcam', img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

canvas = tk.Canvas(root, width=600, height=300)
canvas.grid(columnspan=3, rowspan=3)

button = tk.Button(root, text="Take Attendance", command=show, bg="#5caf47", fg="White", height=2, width=15)
button.grid(columnspan=3, row=2)

image = Image.open('F:\Attendance\Attendance\Face Attendance\icon.png')
logo = image.resize((150,150),Image.ANTIALIAS)
logo= ImageTk.PhotoImage(logo)
logo_lable = tk.Label(image=logo)
logo_lable.image = logo
logo_lable.grid(column=1, row=1)


root.mainloop()

