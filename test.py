import numpy as np
import face_recognition
import os
from datetime import datetime
import cv2
import tkinter as tk
from PIL import Image, ImageTk
#These lines import various libraries and modules,
#including NumPy for numerical operations,
#face_recognition for face recognition,
#os for working with the operating system,
#datetime for handling date and time,
#cv2 for computer vision, and
#tkinter for creating the graphical user interface (GUI).
# The PIL library is used for handling images.

#initiallize a Tkinter root object
root = tk.Tk()
#This line initializes a Tkinter root object, which is the main window for the graphical user interface.

# from PIL import ImageGrab
path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
#These lines define a directory path path where images for face recognition are stored,
# and create empty lists images and classNames to store the loaded images and their corresponding names.

print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)
#This loop iterates through the files in the specified directory, reads each image using OpenCV (cv2),
# and appends the image to the images list. It also extracts the base name (without extension) of the file and
# appends it to the classNames list.

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(img)

        if len(faces) > 0:
            encode = face_recognition.face_encodings(img, faces)[0]
            encodeList.append(encode)

    return encodeList
#This function findEncodings takes a list of images as input and returns a list of face encodings for those images.
# It converts the image from BGR to RGB format and uses face_recognition to extract the face encodings.

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
#This function markAttendance is responsible for marking attendance in a CSV file.
# It reads the existing data from the file, checks if the name is already in the list,
# and if not, it appends the name and the current time to the file.


encodeListKnown = findEncodings(images)
print('Encoding Complete')
#This line calls the findEncodings function to obtain a list of known face encodings for the loaded images.

def show():
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        # img = captureScreen()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            # print(faceDis)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                # print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                markAttendance(name)

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
#This function show initializes the webcam capture, resizes the captured frame,
# detects faces in the frame, performs face recognition, and marks attendance if a recognized face matches a known face.

canvas = tk.Canvas(root, width=600, height=300)
canvas.grid(columnspan=3, rowspan=3)

button = tk.Button(root, text="Take Attendance", command=show, bg="#5caf47", fg="White", height=2, width=15)
button.grid(columnspan=3, row=2)
#These lines create a Tkinter canvas and a button.
# The button triggers the show function when clicked and is styled with a green background and white text.
image = Image.open('icon.png')
logo = image.resize((150,150),Image.ANTIALIAS)
logo= ImageTk.PhotoImage(logo)
logo_lable = tk.Label(image=logo)
logo_lable.image = logo
logo_lable.grid(column=1, row=1)
#This code loads an image named 'icon.png,' resizes it, and displays it in the Tkinter window.

root.mainloop()
#This line starts the main loop of the Tkinter application, allowing the graphical interface to run.
