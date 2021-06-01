from flask import Blueprint, request, jsonify
import os
import base64
from io import BytesIO
from PIL import Image
import face_recognition_test
import sleep_test
#from yawpitchraw import return_std_model, train_model, detect_face_points, compute_features
import yawpitchraw
import tensorflow as tf
from keras.models import load_model
import pickle
import boto3
from botocore.exceptions import NoCredentialsError
from werkzeug.utils import secure_filename
import face_recognition_train

AWS_ACCESS_KEY_ID = "AKIAZDET2MTHTCIVUF2R"
AWS_SECRET_ACCESS_KEY = "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"
AWS_DEFAULT_REGION = "ap-northeast-2"

client_s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID,
                         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                         region_name=AWS_DEFAULT_REGION)


bp = Blueprint('main', __name__, url_prefix='/')
# main_model = None
# main_eye_model = None
# main_ypr_model = None
# main_face_model = None

resource_s3 = boto3.resource('s3', aws_access_key_id= AWS_ACCESS_KEY_ID, aws_secret_access_key = AWS_SECRET_ACCESS_KEY)
client_s3 = boto3.client('s3', aws_access_key_id= AWS_ACCESS_KEY_ID, aws_secret_access_key = AWS_SECRET_ACCESS_KEY)

train_directory_by_group = 'C:/'

print("tensorflow version ")
print("tensor version : " + str(tf.__version__))

def call_Model():
    global main_model
    #main_model = s3.Object('capstonefaceimg', 'load/facenet_keras.h5')
    main_model= tf.keras.models.load_model("facenet_keras.h5")

def call_eye_Model():
    global main_eye_model
    #main_eye_model = load_model(resource_s3.Object('capstonefaceimg','load/2021_05_19_05_31_31.h5'))
    main_eye_model = load_model('2021_05_19_05_31_31.h5')


def call_ypr_Model():
    global main_ypr_model
    #main_ypr_model = s3.Object('capstonefaceimg','load/model.h5')
    #main_ypr_model = load_model(resource_s3.Object('capstonefaceimg','load/model.h5'))
    main_ypr_model = load_model('model.h5')

def call_face_Model():
    global main_face_model
    #main_face_model = pickle.load(open(resource_s3.Object('capstonefaceimg','load/finalized_model.h5'), 'rb'))
    #main_face_model = s3.Object('capstonefaceimg','load/finalized_model.h5')
    main_face_model = pickle.load(open('finalized_model.h5','rb'))


@bp.route('/')
def hello_pybo2():
    return 'Hello, pybo2!!'

@bp.route('/focusInit')
def create_focus_folder():
    focus_main_directory = "C:/FocusHawkEyeMain"
    try:
        if not os.path.exists(focus_main_directory):
            os.makedirs(focus_main_directory)  # 디렉토리 생성 / 그룹 폴더 생성
            print("=== FocusHawkEyeMain folder created ===")
    except OSError:
        print("=== FocusHawkEyeMain folder already exists ===")  # 이미 생성된 폴더의 경우 다음으로 넘어간다

    # 웹캠 캡쳐 후 test 이미지가 들어갈 경로
    focus_capture_image_directory = focus_main_directory + '/webCamCapture/temp'
    try:
        if not os.path.exists(focus_capture_image_directory):
            os.makedirs(focus_capture_image_directory)  # 디렉토리 생성 / 웹캠 캡쳐 이미지 폴더 생성
            print("=== FocusHawkEye WebCam Capture folder created ===")
    except OSError:
        print("=== FocusHawkEye WebCam Capture folder already exists ===")  # 이미 생성된 폴더의 경우 다음으로 넘어간다

    # face recognition에서 npz 파일 저장 될 경로
    focus_npz_directory = focus_main_directory + '/npzSave'
    try:
        if not os.path.exists(focus_npz_directory):
            os.makedirs(focus_npz_directory)  # 디렉토리 생성 / npz 파일 저장 폴더 생성
            print("=== FocusHawkEye npz file save folder created ===")
    except OSError:
        print("=== FocusHawkEye npz file save folder already exists ===")  # 이미 생성된 폴더의 경우 다음으로 넘어간다

    return 'folder create success'


@bp.route('/load/model')
def loadModel():
    call_Model()
    print('call model complete')
    call_ypr_Model()
    print('call ypr_model complete')
    call_face_Model()
    print('call face_model complete')
    call_eye_Model()
    print('call eye_model complete')

    return 'load model success'


@bp.route('/urlTest')
def index():
    return 'hihihihi'

# 웹캠 캡쳐 후 넘어온 이미지로 테스트 수행 & 결과 값 리턴
@bp.route('/image', methods=['GET', 'POST'])
def testGetImage():


  file = request.form['file']
  starter = file.find(',')
  image_data = file[starter + 1:]
  image_data = bytes(image_data, encoding="ascii")

  im = Image.open(BytesIO(base64.b64decode(image_data)))

  #인식결과
  face_recognition_result = False

  #졸음 결과
  sleep_result = False


  userID = request.form['userId']
  groupID = request.form['groupId']

  #print(str('userId' + str(userID)))
  #print(str('groupId') + str(groupID))


  print(im)

  # 기존 코드
  #im.save(os.path.join("G:/내 드라이브/capstone_2/data/5-celebrity-faces-dataset/val/temp","test.jpg"))
  im.save(os.path.join("C:/FocusHawkEyeMain/webCamCapture/temp", "test.jpg"))  # focus 전용 폴더에 저장
  #위에 여기바꿔주세요~~~~~~~~~~~~


  #im.save(s3.Object('capstonefaceimg', 'load/data/5-celebrity-faces-dataset/val/temp","test.jpg'))
  #client_s3.meta.client.upload_file('load/data/5-celebrity-faces-dataset/val/temp/test.jpg', 'capstonefaceimg', im)

  # 그룹, 유저 정보 받아와서 각각 디렉토리 생성하기
  #groupId = 1
  #userId = 1

  # 저장 경로 생성
  #currentPath = "G:/내 드라이브/capstone_2/data/5-celebrity-faces-dataset/val/temp" + "/webcam/group" + str(groupId) + "/user" + str(userId)
  #currentPath = "G:/내 드라이브/capstone_2/data/5-celebrity-faces-dataset/val/temp"
 # 이미 존재하는 경로인지 검사
 #  try:
 #      if not os.path.exists(currentPath):
 #          os.makedirs(currentPath) # 디렉토리 생성 / 그룹 폴더 생성하고 내부에 유저 폴더 추가 생성
 #  except OSError:
 #      print("Error : Cannot create directory")  # 이미 생성된 폴더의 경우 이미지만 변경

  #im.save(os.path.join(currentPath, "test.jpg")) #경로에 test란 이름으로 이미지 저장

  ## 분석?

  final_recognition_score = face_recognition_test.return_score(main_model, main_face_model)
  final_sleep_score = sleep_test.return_sleep_score(main_eye_model)
  final_yaw_pitch_role_score = yawpitchraw.return_ypr_score(main_ypr_model)

  final_recognition_userID = final_recognition_score[-1]
  print("usrID : " + str(final_recognition_userID))

  # 인식 결과로 알려준 유저 넘버와 테스트 이미지를 보낸 유저 넘버 일치 시 True
  if final_recognition_userID == userID:
      face_recognition_result = True

  # 두 눈중 하나라도 음수가 나온다면 True를 준다
  if str(final_sleep_score[0]).find('-') != -1  or str(final_sleep_score[1]).find('-')!= -1:
      sleep_result = True


  return jsonify({
      "name" : final_recognition_score,
      "sleepLeft" : str(final_sleep_score[0]),
      "sleepRight" : str(final_sleep_score[1]),
      "yaw" : str(final_yaw_pitch_role_score[0]),
      "pitch" : str(final_yaw_pitch_role_score[1]),
      "roll" : str(final_yaw_pitch_role_score[2]),
      "attendance" : face_recognition_result,
      "sleepResult" : sleep_result
  })

# @bp.route('/imageTest', methods=['POST'])
# def sendImageTest():
#
#     f = request.files['file']
#     filename = secure_filename(f.filename)
#     f.save(os.path.join("C:/FocusHawkEyeMain", filename))
#
#     return 'image test success'



# Train 이미지 구글 드라이브에 디렉토리에 맞춰서 저장    /   기존의 finalized_model 사용 / finalized 모델은 따로 로컬에서 만드는 모습 보여주기
@bp.route('/groupImages', methods=['POST'])
def getTrainImage():

    req = request.json  # json parsing

    userNum = len(req['groupData'])  # = 유저 수
    groupName = req['groupData'][0]['groupName']  # = 그룹 이름 / 그룹 이름은 모두 공통

    # 테스트 start
    print(str(userNum))  # 현재 그룹 내부 유저 수
    print(str(groupName)) # 그룹 이름
    # 테스트 end


    # 구글드라이브 폴더 테스트
    #s3.meta.client.upload_file('load/data/5-celebrity-faces-dataset/train', 'capstonefaceimg')
    #gdTestPath = "G:/내 드라이브/capstone_2/data/5-celebrity-faces-dataset/train"  # 그룹이 추가되는 경로

    gdTestPath = "C:/FocusHawkEyeMain/train"  # 수업 전 유저 별 트레인 이미지 저장되는 경로

    # 그룹도 나눠야 하기에 그룹 별로 경로 나누기 추가
    train_directory_by_group = gdTestPath + '/' + groupName    # 그룹이름으로 할것인지??

    #bucket_name = "capstonefaceimg"
    #directory_name = "load/data/5-celebrity-faces-dataset/train"  # it's name of your folders
    #client_s3.put_object(Bucket=bucket_name, Key=(directory_name + '/'))

    # 그룹이 추가되는 경로
    #client_s3.put_object(Bucket=bucket_name, Key=(directory_name + '/'))   # S3 디렉토리 생성 코드

    #gdTestPath = "https://capstonefaceimg.s3.ap-northeast-2.amazonaws.com/load/data/5-celebrity-faces-dataset/train/"
    #gdTestPath = "G:/내 드라이브/capstone_2/data/5-celebrity-faces-dataset/train"  # 그룹이 추가되는 경로
    #gdTestPath = "G:/내 드라이브/capstone_2/data/5-celebrity-faces-dataset/train"  # 그룹이 추가되는 경로

    #위에여기바꿔주세요 이거는 S3에서 가져오는겁니다요~~~~~~~~~~~~~

    #gdTestPath_group = gdTestPath + "/" + str(groupName)     # 그룹 경로 추가

    ### 이미 존재하는 경로인지 검사 ###
    try:
         if not os.path.exists(gdTestPath):
             os.makedirs(gdTestPath)  # 디렉토리 생성 / 트레인 이미지 들어갈 예정
    except OSError:
         print("Error : Cannot create group directory")  # 이미 생성된 폴더의 경우 다음으로 넘어간다

    # try:
    #     if not (gdTestPath_group):
    #         os.makedirs(gdTestPath_group)  # 디렉토리 생성 / 그룹 폴더 생성
    # except OSError:
    #     print("Error : Cannot create group directory")  # 이미 생성된 폴더의 경우 다음으로 넘어간다



   ### 유저 아이디 별 디렉토리 생성하고 각각 이미지 저장 ###
    for i in range(userNum):
        userId = req['groupData'][i]['userId']
        #print(str(userId))
        gdTestPath_group_user = gdTestPath + "/user" + str(userId) # 그룹 내 유저별 경로 생성
        #gdTestPath_group_user = directory_name + '/' + str(groupName) + "/user" + str(userId)  # 그룹 내 유저별 경로 생성

        #client_s3.put_object(Bucket=bucket_name, Key=(gdTestPath_group_user + '/'))  # S3 디렉토리 생성

        try:
             if not os.path.exists(gdTestPath_group_user):   # 유저 경로 중복 검사
                 os.makedirs(gdTestPath_group_user) # 중복x -> 디렉토리 생성
        except OSError:
             print("Error : Cannot create user" + str(userId) + " directory")

        userImageNum = len(req['groupData'][i]['images'])  # 현재 유저 별 이미지 수가 달라서 따로 이미지 개수 변수 선언

        ## 각 유저 경로에 이미지 각각 등록되어있는 개수 만큼 저장
        for k in range(userImageNum):
            userImage = req['groupData'][i]['images'][k]
            starter = userImage.find(',')
            userImageConvert = bytes(userImage[starter + 1:], encoding = "ascii")
            im = Image.open(BytesIO(base64.b64decode(userImageConvert)))
            im.save(os.path.join(gdTestPath_group_user, "user" + str(userId) + "TrainImage" + str(k+1) +".jpg"))

        #      # 생성한 S3 경로에 트레인 이미지 저장
        #      client_s3.meta.client.upload_file(gdTestPath_group_user + '/user' + str(k+1)+'.jpg',
        #                                        'capstonefaceimg', im)

        # 트레인 시작
        #face_recognition_train.face_recognition_training()

    return str(req['groupData'][0]['groupName'])

    ######################################################


# Group member Train 요청
@bp.route('/train', methods=['GET'])
def memberTrain():

    # 트레인 시작
    face_recognition_train.face_recognition_training()

    return 'train success!'


# 회원가입 시 해당 유저의 이메일과 트레인용 이미지 3장 받기
@bp.route('/send/train/image', methods=['POST'])
def getTrainImageForUserRegister():

    req = request.json  # json parsing

    # userEmail = req['groupData']  # 유저 이메일
    # groupName = req['groupData'][0]['groupName']  # = 그룹 이름 / 그룹 이름은 모두 공통

    return 'send 3 images success!!'


