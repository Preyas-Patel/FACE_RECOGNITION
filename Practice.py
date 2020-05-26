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
import mysql.connector

# def update_att():
#     mydb = mysql.connector.connect(host="localhost", user="root", passwd="43890", db="mydatabase")
#     myCursor = mydb.cursor()
#     myCursor.execute("SELECT * FROM attendance")
#     records = myCursor.fetchall()
#     length_db = len(records)
#
#     if length_db == 0:
#         df = pd.read_json("attendance.json")
#         length_df = len(df.columns)
#         for i in range(length_df):
#             Id = (df[i].Id)
#             date = (df[i].Date)
#             time = (df[i].Time)
#             myCursor.execute(""" INSERT INTO attendance(Id,date1,time1,att,totclass) VALUE    S %s,%s,%s,%s,%s)""",
#                              (Id, date, time, 0, 0))
#
#             col_names = ['ID', 'Date', 'Time']
#             attendance = pd.DataFrame(columns=col_names)
#             for i in range(1700000, 1710000, 1):
#                 Id = i
#                 date = 0
#                 time = 0
#                 attendance.loc[len(attendance)] = [Id, date, time]
#             fileName = "Attendance/classtest.json"
#             attendance.to_json(fileName, orient="index")
#
#         else:
#
#             df = pd.read_json("Attendance/in.json")
#             length_df = len(df.columns)
#             date_json = (df[0].Date)
#             check = 0
#             for row in records:
#                 date_db = row[1]
#                 if (date_db == date_json):
#                     check = 1
#                     break
#                 else:
#                     check = 0
#             if (check == 1):
#                 pass
#             else:
#                 myCursor.execute("UPDATE attendance SET totclass=totclass+1")
#
#             for i in range(length_df):
#                 id_json = (df[i].ID)
#                 date_json = (df[i].Date)
#                 time_json = (df[i].Time)
#                 date = str(datetime.date.today())
#                 for row in records:
#                     id_db = row[0]
#                     date_db = row[1]
#                     if (id_json == id_db and date_db != date_json):
#                         sql = " UPDATE attendance SET date1=%s,time1=%s,att=att+1 WHERE id=%s"
#                         val = (date_json, time_json, id_json)
#                         myCursor.execute(sql, val)
#
#                 mydb.commit()
#                 mydb.close()


# sql = "CREATE TABLE " + DB_table_name + """
#                         (ID INT NOT NULL AUTO_INCREMENT,
#                          ENROLLMENT varchar(100) NOT NULL,
#                          NAME VARCHAR(50) NOT NULL,
#                          DATE VARCHAR(20) NOT NULL,
#                          TIME VARCHAR(20) NOT NULL,
#                              PRIMARY KEY (ID)
#                              );
window = Tk()
window.geometry('1000x800')
window.configure(bg='grey15')
window.resizable(width=False, height=False)
window.title("Tech Giant Attendance System")
# window.configure(background='#D0D3D4')
image = PIL.Image.open("logo.png")
photo = PIL.ImageTk.PhotoImage(image)
lab = Label(image=photo, bg='grey15')
lab.pack()

fn = StringVar()
entry_name = Entry(window, textvar=fn, width=22, font=("roboto", 15))
entry_name.place(x=265, y=260)
ln = StringVar()
entry_id = Entry(window, textvar=ln, width=22, font=("roboto", 15))
entry_id.place(x=265, y=318)
em = StringVar()
enter_email = Entry(window, textvar=em, width=22, font=("roboto", 15))
enter_email.place(x=265, y=375)


def clear1():
    entry_name.delete(first=0, last=22)


def clear2():
    entry_id.delete(first=0, last=22)


def clear3():
    enter_email.delete(first=0, last=22)


def close():
    quit()


# Id=ln.get()
# name=fn.get()

def detect():
    Id = ln.get()
    name = fn.get()
    email = em.get()
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
    row = [Id, name, email]
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
    # harcascadePath = "haarcascade_frontal.xml"
    # harcascadePath = "haarcascade_frontalface_default.xml"
    # detector = cv2.CascadeClassifier(harcascadePath)
    faces, Id = ImagesAndNames("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save(r"C:\Users\Prey\PycharmProjects\attedance\Trainner.yml")


# dn = StringVar()
# # entry_name_del = Entry(window, textvar=dn)
# # entry_name_del.place(x=150, y=507)


def track_user():
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
    recognizer.read("Trainner.yml")

    faceCascade = cv2.CascadeClassifier(r"C:\Users\Prey\PycharmProjects\attedance\haarcascade_frontal.xml")
    df = pd.read_csv(r"C:\Users\Prey\PycharmProjects\attedance\StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # col_names = ['Id', 'Name', 'Date', 'Time']
    # attendance = pd.DataFrame(columns=col_names)
    # fileName = "attendance.csv"
    # attendance.to_csv(fileName, index=False)
    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if conf < 60:
                with open("attendance.csv", 'a') as f:
                    ts = time.time()
                    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                    aa = df.loc[df['Id'] == Id]['Name'].values
                    tt = str(Id) + "-" + aa
                    z = [Id, aa, date, timeStamp]
                    writer = csv.writer(f)
                    writer.writerow(z)

            else:
                Id = 'Unknown'
                tt = str(Id)
            if conf > 75:
                noOfFile = len(os.listdir("ImagesUnknown")) + 1
                cv2.imwrite("ImagesUnknown\Image" + str(noOfFile) + ".jpg", im[y:y + h, x:x + w])
            cv2.putText(im, str(tt), (x, y + h), font, 1, (255, 255, 255), 2)
        # attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
        cv2.imshow('im', im)
        if (cv2.waitKey(1) == ord('q')):
            break

    cam.release()
    cv2.destroyAllWindows()
    # while True:
    #     ret, im = cam.read()
    #     gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #     faces = faceCascade.detectMultiScale(gray, 1.2, 5)
    #     for (x, y, w, h) in faces:
    #         cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
    #         Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
    #         if (conf < 70):
    #             with open("attendance.csv", 'a') as f:
    #
    #                 ts = time.time()
    #                 date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    #                 timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    #                 aa = df.loc[df['Id'] == Id]['name'].values
    #                 tt = str(Id) + "-"
    #                 #fields = [Id, date, timeStamp]
    #                 attendance.loc[len(attendance)] = [Id, aa, date, timeStamp]
    #                 writer = csv.writer(f)
    #                 writer.writerow(attendance)
    #
    #
    #
    #
    #         else:
    #             Id = 'Unknown'
    #             tt = str(Id)
    #         if (conf > 75):
    #             noOfFile = len(os.listdir("ImagesUnknown")) + 1
    #             cv2.imwrite("ImagesUnknown\Image" + str(noOfFile) + ".jpg", im[y:y + h, x:x + w])
    #         cv2.putText(im, str(tt), (x, y + h), font, 1, (255, 255, 255), 2)
    #     #attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
    #     #df = pd.read_csv("attendance.csv", sep=",")
    #     #df.drop_duplicates(subset=['Id'], inplace=True, keep='first')
    #     cv2.imshow('im', im)
    #     if (cv2.waitKey(1) == ord('q')):
    #         break
    # # ts = time.time()
    # # date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    # # timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    # # Hour, Minute, Second = timeStamp.split(":")
    # fileName = "attendance.csv"
    # #attendance.to_csv(fileName, index=False)
    # cam.release()
    # cv2.destroyAllWindows()


def update():
    df = pd.read_csv("attendance.csv")
    df.drop_duplicates(subset=['Id'], inplace=True, keep='first')
    df.to_csv('final.csv', index=False)


label1 = Label(window, text="Create & Develope By Tech Giant", fg='DeepSkyBlue2', bg='grey15',
               font=("roboto", 20, 'bold')).place(x=280, y=150)
label2 = Label(window, text="New User", fg='#717D7E', bg='grey15', font=("roboto", 25, "bold")).place(x=20, y=200)
label3 = Label(window, text="Enter Name :", fg='black', bg='grey15', font=("roboto", 18)).place(x=20, y=260)
label4 = Label(window, text="Enter Roll Number :", fg='black', bg='grey15', font=("roboto", 18)).place(x=20, y=315)
label5 = Label(window, text="Enter Email address :", fg='black', bg='grey15', font=("roboto", 18)).place(x=20, y=370)

# heading = Label(window, text="Create & Developed By tecch Giant", fg="grey15", )
# status=Label(window,textvariable=v,fg='red',bg='#D0D3D4',font=("roboto",15,"italic")).place(x=20,y=150)
label6 = Label(window, text="Already a User?", fg='#717D7E', bg='grey15', font=("roboto", 25, "bold")).place(x=20,
                                                                                                             y=600)
# label7 = Label(window, text="Delete a users information", fg='#717D7E', bg='grey15',
#               font=("roboto", 20, "bold")).place(x=20, y=450)
# label8 = Label(window, text="Enter Id :", fg='black', bg='grey15', font=("roboto", 15)).place(x=20, y=500)

button1 = Button(window, text="Exit", width=5, fg='#000000', bg='Red', relief=RAISED, font=("roboto", 15, "bold"),
                 command=exit)
button1.place(x=870, y=740)
button2 = Button(window, text="Submit", width=5, fg='#000000', bg='dark green', relief=RAISED,
                 font=("roboto", 15, "bold"),
                 command=detect, height=1)
button2.place(x=20, y=450)
button3 = Button(window, text="Train Images", fg='#000000', bg='dark green', relief=RAISED, font=("roboto", 15, "bold"),
                 command=train_image)
button3.place(x=20, y=530)
button4 = Button(window, text="Track User", fg='#000000', bg='dark green', relief=RAISED, font=("roboto", 15, "bold"),
                 command=track_user)
button4.place(x=20, y=680)
button5 = Button(window, text="clear", fg='#000000', bg='red', relief=RAISED, font=("roboto", 15, "bold"),
                 command=clear1)
button5.place(x=570, y=255)
button6 = Button(window, text="clear", fg='#000000', bg='red', relief=RAISED, font=("roboto", 15, "bold"),
                 command=clear2)
button6.place(x=570, y=313)
button7 = Button(window, text="clear", fg='#000000', bg='red', relief=RAISED, font=("roboto", 15, "bold"),
                 command=clear3)
button7.place(x=570, y=370)
button7 = Button(window, text="clear", fg='#000000', bg='red', relief=RAISED, font=("roboto", 15, "bold"),
                 command=clear3)
button7.place(x=570, y=370)
button8 = Button(window, text="update", fg='#000000', bg='dark green', relief=RAISED, font=("roboto", 15, "bold"),
                 command=update)
button8.place(x=160, y=680)
# button6 = Button(window, text="Delete User", fg='#fff', bg='#8E44AD', relief=RAISED, font=("roboto", 15, "bold"))
# button6.place(x=20, y=550)
window.mainloop()
