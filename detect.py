# YOLO-based balloon detection (educational project)
# Model weights are not included in this repository

import os
import cv2
from ultralytics import YOLO

MODEL_PATH = "best.pt"#trained weights are not included

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError("Model didn't find. Please provide a trained YOLO .pt file.")

model = YOLO(MODEL_PATH) #user should provide trained weights

# open camera (0)
cap = cv2.VideoCapture(0)

#looking first frame for coordinates
success,frame = cap.read()
if success:
    #Taking coordinates of frame
    #and finding middle of screen(frame)
    height, width, _ = frame.shape
    screen_cx= width // 2 #int(width/2) also works
    screen_cy= height // 2



while cap.isOpened():
    # read() turn e tuple first(if we get image) second(image)
    success, frame = cap.read()

    #For now min distance is high because it will change first random ballon then nearest one
    #and nearest ballon's coordinate is none it will change when we found
    near=640
    near_cx , near_cy =None, None

    if success:

        #for mirror effect
        #cv2.flip(frame , flipcode)
        #1 = mirror effect
        #0 = upside down
        frame = cv2.flip(frame, 1)

        #Coordinates of ROI(region of interest)
        roi_y1, roi_y2 = 100, 400
        roi_x1, roi_x2 = 150, 500

        #Getting roi of frame
        #frame[y1:y2, x1:x2] 
        roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        # roi_frame: get this image not all frame
        # stream=True: don't hold in ram use and throw
        # conf=0.4: if you are not sure 40% don't show it
        # classes=[0,2,4] just write if its class ıd any of them
        results = model(roi_frame, stream=True, conf=0.4,classes=[0,2,4])

        #Drawing and naming region on frame
        cv2.rectangle(frame, (roi_x1,roi_y1), (roi_x2,roi_y2), (0,255,0), 1)
        cv2.putText(frame, "TARANACAK ALAN", (roi_x1,roi_y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        for r in results:
            #Access detected object bounding boxes for the current frame
            boxes=r.boxes
            for box in boxes:

                #Take coordinates of top left(x1,y1) bottom right(x2,y2)
                #BUT this coordinates are roi's not general frame
                x1, y1, x2, y2 =box.xyxy[0]
                x1, y1, x2, y2= int(x1), int(y1), int(x2), int(y2)

                #Adding roi_xy for getting real xy
                x1, x2 = x1+roi_x1, x2+roi_x1
                y1, y2 = y1+roi_y1, y2+roi_y1

                #Calculating centers
                cx=int((x1+x2)/2)
                cy=int((y1+y2)/2)

                #Calculating the distance of the object from the center
                dist_cx = cx-screen_cx
                dist_cy = -(cy-screen_cy)

                current_dist=(dist_cx**2 + dist_cy**2)**0.5 #for drawing line to nearest

                #Look box's id and get class name
                ballon_name=model.names[int(box.cls[0])]
                
                #confidence score
                conf= box.conf[0]

                #Drawing rectangle
                #cv2.rectangle(frame, (top left), (bottom right), (color but BGR),(thickness))
                #255,0,0 Blue
                #0,255,0 Green
                #0,0,255 Red
                #-1 thickness means fill inside
                cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,255), 2) #Magenta

                #Drawing a circle middle
                #cv2.circle(frame , (center),(radius), (color), (thickness))
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

                #Show as text
                #cv2.putText(frame, "text", (text's start coordinate), (font), (scale), (color))
                #conf*100:.0f means take zero digit after ,
                text=f"%{conf*100:.0f} {ballon_name}: {dist_cx}, {dist_cy} "
                cv2.putText(frame,text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

                if current_dist< near: #we found nearest so taking coordinates
                    near = current_dist
                    near_cx = cx
                    near_cy = cy

        #if we found nearest ballon draw a line
        if near_cx != None:
            cv2.line(frame, (screen_cx,screen_cy), (near_cx,near_cy), (255,255,0), 1, cv2.LINE_AA)

        #Cross at middle of screen
        length=10
        #cv2.line(frame, (start coordinate), (end coordinate), (color), (thickness))
        cv2.line(frame, (screen_cx-length,screen_cy), (screen_cx+length,screen_cy), (0,0,255), 1) #Horizontal
        cv2.line(frame, (screen_cx,screen_cy-length), (screen_cx,screen_cy+length), (0,0,255), 1) #Vertical

        #show at window with this name
        cv2.imshow("Balon Takip Sistemi",frame)

        # press 'q' to quit 
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break


cap.release() # close the camera
cv2.destroyAllWindows() # close all windows


#Class ID for this data
# {0: 'KirmiziBalon', 1: 'KirmiziPatlamisBalon', 2: 'MaviBalon', 3: 'MaviPatlamisBalon', 4: 'YesilBalon', 5: 'YesilPatlamisBalon'}
