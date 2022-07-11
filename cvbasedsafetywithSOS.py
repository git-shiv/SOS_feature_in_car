
import cv2
import mediapipe as mp
import time
import math
import numpy as np
from twilio.rest import Client
#################################################
#Credentials for SOS### login to twilio and get your own credentials and paste here
account_sid = "***********************************"
auth_token = "*********************************"
number = "+***********" # from which msg will be send , provided by twilio
################################################
##########
#SOS MSG # your custom msg
msg = "HELLO,XYZ, Your Driver is Unconsious. Kindly report"
msg1 = "Health issue"
msg2 = "technical failure"
class handTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5,modelComplexity=1,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils


    def handsFinder(self,image,draw=True):
        imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image


    def findPosition(self, img, handNo=0, draw=True):
        
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
            # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                # print(id, cx, cy) 
            
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return self.lmList



class PoseDetector():

    def __init__(self,mode = False,model_complexity = 1, smooth_lm = True, segmentEnable = True, detectionCon = 0.5,TrackeCon = 0.5):

        self.mode = mode
        self.segmentEnable = segmentEnable
        self.model_complexity = model_complexity
        self.smooth_lm = smooth_lm
        self.detectionCon = detectionCon
        self.TrackeCon = TrackeCon

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.mode,self.model_complexity,self.smooth_lm,self.segmentEnable,self.detectionCon,self.TrackeCon)


        
    def findPose(self,image,draw):
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(image)
        
    
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if draw ==  True:
            self.mp_drawing.draw_landmarks(
                image,
                self.results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec = self.mp_drawing_styles.get_default_pose_landmarks_style())
        return image
    
    def find_marks(self,image):
        marks = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = image.shape
            
                cx, cy = int(lm.x * w), int(lm.y * h)
                marks.append([id,cx,cy])
        return marks
    
def showfps(frame):
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    # Flip the image horizontally for a selfie-view display.
    cv2.putText(frame,str(int(fps)),(70,50),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,255),4)

def SOS(msg,toNo):
    client = Client(account_sid,auth_token)
    message =  client.api.account.messages.create(
    to = toNo,
    from_= number,
    body = msg
    ) 

def gestureTrack(tipids,marks):
    # this function tells the tips are opened or not of fingers
    fingUp = [] 
    if marks[tipids[0]][1] < marks[tipids[0] - 1][1]:
            fingUp.append(1)
    else:
            fingUp.append(0)
    for id in range(1, 5):
            if marks[tipids[id]][2] < marks[tipids[id] - 2][2]:
                fingUp.append(1)
            else:
                fingUp.append(0)

    return fingUp




def main():
    SOS("Hi,XYZ...Safety Systems are Now active","+91*******") # second argument is mobile no. on which you want to send message
    x = [465,450,425,400,390,376,360,330,315,281,269]
    y = [25,30,35,40,45,50,55,60,65,70,75]
    ThresValue = 35  #the min distace acceptable to be a active driver
    coff = np.polyfit(x, y, 2)
    ptime = 0 
    cap = cv2.VideoCapture(0)
    PoseTracker = PoseDetector()
    gesture = handTracker()
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
      
            continue
        
        frame = PoseTracker.findPose(frame,False)
        frame = gesture.handsFinder(frame,draw=False)
        points = PoseTracker.find_marks(frame)
        marks = gesture.findPosition(frame,draw=False)

        if len(marks) != 0:
            # print(marks[4])
            tipids =[4,8,12,16,20] # 4 for thumb,8 for index ,12 for middle, 16 for ring finger,20 for pinky finger
            tips = gestureTrack(tipids,marks)
            # print(tips) # tips is list of fingures which a up [thumb,index finger,middle finger,ring finger,pinky finger]
                        # 0 if it is folded 1 if it is open
            if tips[3] == 0 and tips[4] == 0 and tips[0] == 1:
                print("Technical failure")
                SOS(msg1,"+91**********") #second argument is mobile no. on which you want to send message
            if tips[0] == 0 and tips[3] == 1 and tips[4] == 1:
                print("health issue")
                SOS(msg2,"+91**********") #second argument is mobile no. on which you want to send message
        

        if len(points) != 0:  
            x11,y11 = points[11][1], points[11][2]
            x12,y12 = points[12][1],points[12][2]
        
            dis = math.hypot((x12 - x11),(y12-y11))
            #print(dis)

            cv2.circle(frame,(x11,y11),15,(255,0,228),cv2.FILLED)
            cv2.circle(frame,(x12,y12),15,(255,0,228),cv2.FILLED)

            A, B, C = coff
            disCM = A * dis ** 2 + B * dis + C
            disCM = int(disCM)
            #print(disCM)

            if disCM > ThresValue:
                print("active driver")
                print("  ")
            else:
                print("driver unconscious")
                SOS(msg,"+91**********") #second argument is mobile no. on which you want to send message
        else:
            print("No driver Detetcted")
        # showfps(frame)
       
        cv2.imshow('MediaPipe Pose', cv2.flip(frame, 1))
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break  

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

