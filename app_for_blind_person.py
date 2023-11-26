from flask import Flask , render_template , request
import base64
from PIL import Image
from io import BytesIO
import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
# from cvzone.HandTrackingModule import HandDetector
import numpy as np
# from sound_module2 import *
import threading
import os
from cvzone.HandTrackingModule import HandDetector
import time
import sys

app = Flask(__name__)

last_frame_time = 0
current_frame = None

np.random.seed(20) #type:ignore
class Detector:
    
    def __init__(self,video,config,model,classes,open_cv_image_by_me):
        # self.write_txt()

        self.videoPath = video
        self.configPath = config
        self.modelPath = model
        self.classesPath = classes
        self.display_text_for_object = ''
        self.max_faces = 10
        self.detector = FaceMeshDetector(maxFaces=self.max_faces)
        self.time = 0
        self.objects_names = []
        self.on = False

        self.detector_deaf = HandDetector(staticMode=False, maxHands=4, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)
        # self.hand_signs = {"A":[0,0,0,0,0], "B":[0,1,1,1,1], "D":[0,1,0,0,0], "F":[0,0,1,1,1], "I":[0,0,0,0,1]}
        # self.list3 = list(self.hand_signs.keys())

        self.bb = False
        self.deaf_timer = 5
        self.t_timer = 0
        self.vv = 40
        self.width_of_screen = 1000
        self.height_of_screen = 750

        self.open_cv_image_by_me = open_cv_image_by_me

        
        #######################################
        
        self.net = cv2.dnn_DetectionModel(self.modelPath,self.configPath) # type: ignore
        self.net.setInputSize(320,320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5,127.5,127.5))
        self.net.setInputSwapRB(True)
    
        self.readclasses()
    
    def readclasses(self):
        with open(self.classesPath,'r') as f:
            self.classlist = f.read().splitlines()
        
        self.classlist.insert(0,'__Background__')
        
        self.colorlist = np.random.uniform(low=0,high=255,size=(len(self.classlist),3)) #type:ignore

    def write_txt(self):
        with open('mm.txt','w') as hmm:
                pass
        
    def _threadSoundCALL(self,*ignore):
        try:
            speak(" ".join(ignore)) # type: ignore
        except RuntimeError:
            pass
        print('speaker ',ignore)
    
    def _draw_lines(self,frame,w,h):
        # (x, y, w, h) = cv2.getWindowImageRect("Result")
        cv2.line(frame, (int(w/2),0), (int(w/2),int((h/2)-self.vv)), (0, 255, 255), 5)
        cv2.line(frame, (int(w/2),h), (int(w/2),int((h/2)+self.vv)), (0, 255, 255), 5)
        cv2.line(frame, (0,int(h/2)), (int((w/2)-self.vv),int(h/2)), (0, 255, 255), 5)
        cv2.line(frame, (int(w),int(h/2)), (int((w/2)+self.vv),int(h/2)), (0, 255, 255), 5)
        cv2.rectangle(frame,(int((w/2)-self.vv),int((h/2)-self.vv)),(int((w/2)+self.vv),int((h/2)+self.vv)),(0, 255, 255),5)
    
    def _pos_hint(self,x_,y_,w_,h_,w,h,vv):
        strt = ''
        # if x_ < w/2 and (y_+w_) < h/2:
        #     return "on top-left"
        # elif x_ < w/2 and (y_+w_) <= h:
        #     return "on left"
        if x_ < w/2 and (y_+h_) < h/2:
            if (x_+w_) < w/2:
                strt = "on top-left"
            elif (x_+w_) > w/2:
                strt = "on top"
            # else:
            #     return "in front"
        elif x_ > w/2:
            if (y_+h_) < h/2:
                strt = "on top-right"
            elif (y_+h_) > h/2:
                if y_ > h/2:
                    strt = "on bottom-right"
                elif y_ < h/2:
                    strt = "on right"
                # else:
                #     return "in front"
            # else:
            #     return "in front"
        elif x_ < w/2 and (y_+h_) > h/2:
            if y_ > h/2:
                if (x_+w_) < w/2:
                    strt = "on bottom-left"
                elif (x_+w_) > w/2:
                    strt = "on bottom"
                # else:
                #     return "in front"
            elif y_ < h/2:
                strt = "on left"
            # else:
            #     return "in front"
        # else:
        #     return "in front"
        
        if x_ < w/2 and y_ < h/2:
            if (x_+w_) > w/2 and (y_+h_) > h/2:
                strt = "in front"
        
        return strt
        
            
    
    def onvideo(self):
            # for i in self.objects_names:
            #     load_sound(i)
        success = True
        # cap = cv2.VideoCapture(0)
        # if cap.isOpened() == False:
        #     print('Error opening file...')
        #     return
        
        # success , img = cap.read()
        img = self.open_cv_image_by_me
        if success:
            try:
                _stop_event = threading.Event()
                self.time += 1
                self.t_timer += 1
                
                img = cv2.resize(img, (self.width_of_screen,self.height_of_screen))
                # self._draw_lines(img,self.width_of_screen,self.height_of_screen)
                
                hands, frame = self.detector_deaf.findHands(img)

                classLabalID , confidentcs , bboxesssssss = self.net.detect(img,confThreshold = 0.4)
                bboxesssssssaa = list(bboxesssssss)
                confidentcs = list(np.array(confidentcs).reshape(1,-1)[0])
                confidentcs = list(map(float,confidentcs))
                
                bboxesssssssaaID = cv2.dnn.NMSBoxes(bboxesssssssaa,confidentcs,score_threshold=0.5,nms_threshold=0.2)
                
                if self.bb == False or self.bb == True:
                    if len(bboxesssssssaaID) != 0:
                        for i in range(0,len(bboxesssssssaaID)):
                            bbox = bboxesssssssaa[np.squeeze(bboxesssssssaaID[i])]
                            classconfidences = confidentcs[np.squeeze(bboxesssssssaaID[i])]
                            classlabelid = np.squeeze(classLabalID[np.squeeze(bboxesssssssaaID[i])])
                            classlebel = self.classlist[classlabelid]
                            classcolor = [int(c) for c in self.colorlist[classlabelid]]
                            if classconfidences>=0.5:
                                desplaytext = "{}:{:.2f}".format(classlebel,classconfidences)
                                self.display_text_for_object = desplaytext
                                if not(classlebel in self.objects_names):
                                    x,y,w,h = bbox
                                    rev = self._pos_hint(x,y,w,h,self.width_of_screen,self.height_of_screen,self.vv)
                                    if rev:
                                        self.objects_names.append(f"{classlebel} {rev},")
                                #print(self.main_disply)
                            x,y,w,h = bbox
                            if self.display_text_for_object != '':
                                cv2.putText(img,self.display_text_for_object,(x,y-10),cv2.FONT_HERSHEY_PLAIN,1,classcolor,2)
                    img , faces = self.detector.findFaceMesh(img,draw=False)
                    if faces:
                        face = faces[0]
                        pointleft = face[145]
                        pointright = face[374]
                        cv2.circle(img,pointleft,5,(255,0,0),cv2.FILLED)
                        cv2.circle(img,pointright,5,(0,255,0),cv2.FILLED)
                        cv2.line(img,pointleft,pointright,(0,0,255),3)
                        w , _= self.detector.findDistance(pointleft,pointright) # type: ignore
                        W = 6.3
                        # d = 50
                        # f = (w*d)/W
                        # print(f)
                        f = 1488
                        d = (W*f)/w
                        d_int = int(d)
                        #cvzone.putTextRect(img,f"Distance: {d_int}cm",(face[10][0]-150,face[10][1]-50),2)
                        if d_int <= 50:
                            self.objects_names.append("Person Too close,")
                            #print("Too close")
                        elif d_int <= 120:
                            self.objects_names.append("Person at Normal distant,")
                            #print("Normal distant")
                        elif d_int <= 200:
                            self.objects_names.append("Person at Far distant,")
                            #print("Far distant")
                        else:
                            self.objects_names.append("Person Too far away,")
                            #print("Too far away")
                
                if self.bb == False:
                    if len(hands) == 1:
                        hand2 = hands[0]
                        lmList3 = hand2["lmList"]
                        fingers4 = self.detector_deaf.fingersUp(hand2)
                        if fingers4 == [1,1,1,1,1]:
                            self.bb = True
                            # print('starting')

                # if self.bb == True:
                #     #print("started"+str(self.time))
                #     if self.time != -1:
                #         if len(hands) == 1:
                #             hand1 = hands[0]
                #             lmList1 = hand1["lmList"]
                #             fingers1 = self.detector_deaf.fingersUp(hand1)
                #             if fingers1 in self.hand_signs.values():
                #                 nn = str(self.list3[list(self.hand_signs.values()).index(fingers1)])
                #                 # print(nn)
                #                 self.objects_names.append(f"Deaf person said {nn},")
                #                 # print('hand_tracked')
                #                 # speak("Sign", list3[list(hand_signs.values()).index(fingers1)])
                #                 # time.sleep(1)
                cv2.imwrite("Result.jpg",img)
                if self.time == self.deaf_timer:
                    self.time = 0
                    if self.objects_names != []:
                        try:
                            
                            # t1 = threading.Thread(target=self._threadSoundCALL,args=self.objects_names)
                            # t1.start()
                            self.on = True
                        except RuntimeError:
                            pass
                with open('mm.txt','w') as hmm:
                    hmm.write('com')
                print(self.objects_names)
                self.objects_names = []
                
                key = cv2.waitKey(1) & 0xff
                if key==ord('q'):
                    _stop_event.set()
                
                # success , img = cap.read()
                img = self.open_cv_image_by_me
                # cv2.destroyAllWindows()
                # sys.exit()
            except Exception as e:
                print(e)



@app.route("/")
def hello_world():
    return render_template('index.html')


def main(open_cv_image_by_):
    try:
        videopath = os.path.join('model_data','test1.mp4')
        configpath = os.path.join('model_data','ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
        modelpath = os.path.join('model_data','frozen_inference_graph.pb')
        classpath = os.path.join('model_data','coco.names')
    
        detector = Detector(videopath,configpath,modelpath,classpath,open_cv_image_by_)
        detector.onvideo()
    except Exception as e:
        print(e)
    # open_cv_image_by = open_cv_image_by_[0]
    # videopath = os.path.join('model_data','test1.mp4')
    # configpath = os.path.join('model_data','ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
    # modelpath = os.path.join('model_data','frozen_inference_graph.pb')
    # classpath = os.path.join('model_data','coco.names')
    
    # detector = Detector(videopath,configpath,modelpath,classpath,open_cv_image_by_)
    # detector.onvideo()


@app.route('/upload', methods=['POST'])
def upload():
    global last_frame_time, current_frame
    image_data = request.form['image']
    img_bytes = base64.b64decode(image_data.split(',')[1])

    # Save the image only if 0.5 seconds have passed
    if time.time() - last_frame_time > 1.0:
        last_frame_time = time.time()
        current_frame = img_bytes

        # Perform your processing logic here
        last_image_data = image_data
        img = Image.open(BytesIO(base64.b64decode(last_image_data.split(',')[1]))).convert('RGB')

        open_cv_image = np.array(img)
    #     # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        # with open('mm.txt','r') as hmm:
        #     if str(hmm.read()) == 'com':
        #         main(open_cv_image)
        #     else:
        #         with open('mm.txt','w') as hmm:
        #             pass
        try:
            main(open_cv_image)
        except Exception as e:
            print(e)
        cv2.imwrite('filename.jpg', open_cv_image)
        # For now, let's print the length of the received image data
        # print("Received image data. Length:", len(img_bytes))
    else:
        current_frame = None

    return 'Image data received successfully'

    # # Get the image data from the POST request
    # img_data = request.form['image']

    # # Convert base64-encoded image data to PIL Image
    # img = Image.open(BytesIO(base64.b64decode(img_data.split(',')[1]))).convert('RGB')

    # open_cv_image = np.array(img)
    # # Convert RGB to BGR
    # open_cv_image = open_cv_image[:, :, ::-1].copy()

    # # Save the image on the server
    # # img.save('uploaded_image.jpg')
    # # cv2.imwrite('captured_img.jpg', open_cv_image)
    # main(open_cv_image)

    # return 'Image successfully uploaded and saved.'

    # global last_image_data

    # # Receive the image data from the client
    # image_data = request.form['image']

    # # If the received image data is not 'None', save it
    # if image_data != 'None':
    #     last_image_data = image_data
    #     img = Image.open(BytesIO(base64.b64decode(last_image_data.split(',')[1]))).convert('RGB')

    #     open_cv_image = np.array(img)
    #     # Convert RGB to BGR
    #     open_cv_image = open_cv_image[:, :, ::-1].copy()
    #     cv2.imwrite('filename.jpg', open_cv_image)

    # return 'Image data received successfully'



# @app.route('/get_last_frame', methods=['GET'])
# def get_last_frame():
#     global last_image_data
#     return last_image_data or 'None'

if __name__=="__main__":
    app.run(host="0.0.0.0") # host="0.0.0.0"
