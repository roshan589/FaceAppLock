import glob
import sys
from pathlib import Path
import cv2
import face_recognition as fr
import numpy as np
import subprocess


def read_file(filepath, image_file_type):
    """ Provide filepaths such as  folder name and file type of images of your database. For Eg: .jpg, .png """
    filepaths = glob.glob(f"{filepath}/*.{image_file_type}")
    return filepaths


def face_filename(filepaths):
    """ FOR IMAGE NAME PLZZ MENTION NAME IN FILE. For example: Roshan.jpg """
    face_names = []
    for name in filepaths:
        face_name = Path(name).stem
        face_names.append(face_name)
    return face_names


def run_app(app_name):
    extension = app_name.split(".")[1]
    print(extension)
    if extension == "exe":
        subprocess.run(app_name)
        sys.exit()
    else:
        with open(app_name, 'r') as file:
            code = file.read()
            exec(code)


def encode_faces(imgs):
    encode_list = []
    for image in imgs:
        img = cv2.imread(image)
        small_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        img_rgb = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
        face_encodings = fr.face_encodings(img_rgb)
        encode_list.append(face_encodings[0])
    return encode_list


def face_recognize(program_name, filepath, filetype="jpg"):
    cap = cv2.VideoCapture(0)

    filepaths = read_file(filepath, filetype)
    encode_known_list = encode_faces(filepaths)

    while True:
        ret, frame = cap.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_cur_frame = fr.face_locations(rgb_frame)
        frame_encode = fr.face_encodings(rgb_frame, face_cur_frame)

        for face_encode, face_loc in zip(frame_encode, face_cur_frame):
            matches = fr.compare_faces(encode_known_list, face_encode)
            face_distance = fr.face_distance(encode_known_list, face_encode)
            match_index = np.argmin(face_distance)
            if matches[match_index]:
                cap.release()
                cv2.destroyAllWindows()
                run_app(program_name)


            else:
                print("Unknown")
        cv2.imshow("Window", frame)
        key = cv2.waitKey(1)

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    face_recognize("mspaint.exe", "images")
