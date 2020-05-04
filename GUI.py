import PIL.Image
import PIL.ImageTk
from tkinter import *
import cv2
import os
import csv
import numpy as np
import PIL.Image
import PIL.ImageTk
import pandas as pd
import datetime
import time

window = Tk()
window.geometry('600x600')
window.resizable(width=False, height=False)
window.title("My Attendance Portal")
window.configure(background='#D0D3D4')
image = PIL.Image.open("logo.png")
photo = PIL.ImageTk.PhotoImage(image)
lab = Label(image=photo, bg='#D0D3D4')
lab.pack()

fn = StringVar()
entry_name = Entry(window, textvar=fn)
entry_name.place(x=150, y=257)
ln = StringVar()
entry_id = Entry(window, textvar=ln)
entry_id.place(x=455, y=257)


def close():
    quit()


# Id=ln.get()
# name=fn.get()

def detect():
    Id = ln.get()
    name = fn.get()
    cascade_face = cv2.CascadeClassifier(r"C:\Users\Prey\PycharmProjects\attedance\haarcascade_frontal.xml")
    cam = cv2.VideoCapture(0)
    img_counter = 0
    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade_face.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Face Registration", frame)
        if not ret:
            break
        k = cv2.waitKey(1)

        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed

            img_name = "{}.jpg".format(name.lower() + "." + Id + '.' + str(img_counter))
            cv2.imwrite("TrainingImage\ " + img_name, roi_gray)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()
    cv2.destroyAllWindows()
    row = [Id, name]
    with open(r"C:\Users\Prey\PycharmProjects\attedance\StudentDetails.csv", 'a+') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    csvFile.close()


def ImagesAndNames(path):
    # get the path of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # create empty face list
    faces = []
    # create empty ID list
    Ids = []
    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # Loading the images in Training images and converting it to gray scale
        g_image = PIL.Image.open(imagePath).convert('L')
        # Now we are converting the PIL image into numpy array
        image_ar = np.array(g_image, 'uint8')
        # getting the Id from the image
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(image_ar)
        Ids.append(Id)
    return faces, Ids


def train_image():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    #harcascadePath = "haarcascade_frontal.xml"
    #harcascadePath = "haarcascade_frontalface_default.xml"
    #detector = cv2.CascadeClassifier(harcascadePath)
    faces, Id = ImagesAndNames("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save(r"C:\Users\Prey\PycharmProjects\attedance\Trainner.yml")


# dn = StringVar()
# # entry_name_del = Entry(window, textvar=dn)
# # entry_name_del.place(x=150, y=507)


def track_user():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("Trainner.yml")
    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(r"C:\Users\Prey\PycharmProjects\attedance\haarcascade_frontal.xml")
    font = cv2.FONT_HERSHEY_SIMPLEX
    df = pd.read_csv(r"C:\Users\Prey\PycharmProjects\attedance\StudentDetails.csv")
    col_names = ['ID', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)
    while (True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            name = df.loc[df['Id'] == Id]['Name'].values
            name_get = str(Id) + "-" + name
            time_s = time.time()
            date = str(datetime.datetime.fromtimestamp(time_s).strftime('%Y-%m-%d'))
            timeStamp = datetime.datetime.fromtimestamp(time_s).strftime('%H:%M:%S')
            attendance.loc[len(attendance)] = [Id, date, timeStamp]
            if (conf > 90):
                noOfFile = len(os.listdir("ImagesUnknown")) + 1
                cv2.imwrite("ImagesUnknown\Image" + str(noOfFile) + ".jpg", img[y:y + h, x:x + w])
                Id = 'Unknown'
                name_get = str(Id)
            cv2.putText(img, str(name_get), (x + w, y + h), font, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
            attendance = attendance.drop_duplicates(keep='first', subset=['ID'])
            fileName = "asd.csv"
            attendance.to_csv(fileName)
            cv2.imshow('img', img)
            if (cv2.waitKey(1) == ord('q')):
                break
            cam.release()
            cv2.destroyAllWindows()



label2 = Label(window, text="New User", fg='#717D7E', bg='#D0D3D4', font=("roboto", 20, "bold")).place(x=20, y=200)
label3 = Label(window, text="Enter Name :", fg='black', bg='#D0D3D4', font=("roboto", 15)).place(x=20, y=250)
label4 = Label(window, text="Enter Roll Number :", fg='black', bg='#D0D3D4', font=("roboto", 15)).place(x=275, y=252)
label5 = Label(window, text="Note : To exit the frame window press 'q'", fg='red', bg='#D0D3D4',
               font=("roboto", 15)).place(x=20, y=100)
# status=Label(window,textvariable=v,fg='red',bg='#D0D3D4',font=("roboto",15,"italic")).place(x=20,y=150)
label6 = Label(window, text="Already a User ?", fg='#717D7E', bg='#D0D3D4', font=("roboto", 20, "bold")).place(x=20,
                                                                                                               y=350)
label7 = Label(window, text="Delete a users information", fg='#717D7E', bg='#D0D3D4',
               font=("roboto", 20, "bold")).place(x=20, y=450)
label8 = Label(window, text="Enter Id :", fg='black', bg='#D0D3D4', font=("roboto", 15)).place(x=20, y=500)

button1 = Button(window, text="Exit", width=5, fg='#fff', bg='red', relief=RAISED, font=("roboto", 15, "bold"),
                 command=exit)
button1.place(x=500, y=550)
button2 = Button(window, text="Submit", width=5, fg='#fff', bg='#27AE60', relief=RAISED, font=("roboto", 15, "bold"),
                 command=detect)
button2.place(x=20, y=300)
button3 = Button(window, text="Train Images", fg='#fff', bg='#5DADE2', relief=RAISED, font=("roboto", 15, "bold"),
                 command=train_image)
button3.place(x=100, y=300)
button4 = Button(window, text="Track User", fg='#fff', bg='#E67E22', relief=RAISED, font=("roboto", 15, "bold"),
                 command=track_user)
button4.place(x=20, y=400)
button6 = Button(window, text="Delete User", fg='#fff', bg='#8E44AD', relief=RAISED, font=("roboto", 15, "bold"))
button6.place(x=20, y=550)
window.mainloop()
