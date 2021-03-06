import cv2, dlib
import numpy as np
from imutils import face_utils
from keras.models import load_model
import os
from keras.preprocessing.image import ImageDataGenerator
import boto3
resource_s3 = boto3.resource('s3')

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

IMG_SIZE = (34, 26)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#predictor = dlib.shape_predictor(resource_s3.Object('capstonefaceimg', "shape_predictor_68_face_landmarks.dat"))
#predictor = dlib.shape_predictor(resource_s3.Object("s3://capstonefaceimg/load/shape_predictor_68_face_landmarks.dat"))



# model.summary()

def crop_eye(img, eye_points):
    x1, y1 = np.amin(eye_points, axis=0)
    x2, y2 = np.amax(eye_points, axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    w = (x2 - x1) * 1.2
    h = w * IMG_SIZE[1] / IMG_SIZE[0]

    margin_x, margin_y = w / 2, h / 2

    min_x, min_y = int(cx - margin_x), int(cy - margin_y)
    max_x, max_y = int(cx + margin_x), int(cy + margin_y)

    eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

    eye_img = img[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

    return eye_img, eye_rect

#img_ori = cv2.imread('G:/내 드라이브/capstone_2/datadata/5-celebrity-faces-dataset/val/temp/test.jpg')
def return_sleep_score(main_eye_model, userEmail):
  print(type(main_eye_model))
  #img_array = np.fromfile('G:/내 드라이브/capstone_2/datadata/5-celebrity-faces-dataset/val/temp/test.jpg', np.uint8)
  img_array = np.fromfile('C:/FocusHawkEyeMain/webCamCapture/temp/'+ userEmail +'/capture/test.jpg', np.uint8)
  #로컬에다 저장하는 애들이에요 val/temp/test 폴더 안에 test.jpg 가져오기



  img_ori = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
  img_ori = cv2.resize(img_ori, dsize=(0, 0), fx=0.5, fy=0.5)

  img = img_ori.copy()
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = detector(gray)

  for face in faces:

    shapes = predictor(gray, face)
    if shapes is not None:
      print('shapes not null')

    shapes = face_utils.shape_to_np(shapes)

    eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
    eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])

    eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
    eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
    eye_img_r = cv2.flip(eye_img_r, flipCode=1)

    eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
    eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.

    pred_l = main_eye_model.predict(eye_input_l)
    pred_r = main_eye_model.predict(eye_input_r)

    # visualize
    state_l = '%.1f' if pred_l > 0.1 else '- %.1f'
    state_r = '%.1f' if pred_r > 0.1 else '- %.1f'

    state_l = state_l % pred_l
    state_r = state_r % pred_r

    # cv2.rectangle(img, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255,255,255), thickness=2)
    # cv2.rectangle(img, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255,255,255), thickness=2)

    # cv2.putText(img, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    # cv2.putText(img, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    sleep_score =[]
    sleep_score.append(state_l)
    sleep_score.append(state_r)

  # cv2.imshow('result', img)
  #print(state_l, state_r)
  return sleep_score

# sleep score가 assign 전에 reference 되는 오류는 사진 인식 및 졸음 인식 모델 성능 문제