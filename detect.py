"""
    Counting people by coral dev mini by GILLES Baptiste
"""
import subprocess
import argparse
import cv2
import os
import time
import numpy as np
import struct
import datetime
from sort import *


from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

#PIN
from periphery import GPIO
import serial


def main():
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    default_labels = 'coco_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_labels))
    parser.add_argument('--top_k', type=int, default=10,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='classifier score threshold')
    args = parser.parse_args()


    #Define in/out put ID and line 
    entered_left_ids = set()  
    exited_left_ids = set()
    entered_right_ids = set()  # Identifiers of objects that crossed the first line
    exited_right_ids = set()  # Identifiers of objects that crossed the second line
    person_counter_enter = 0
    person_counter_leave = 0
    line1_x = 50  
    line2_x = 730
    
    #last send time
    last_send_time = datetime.datetime.now()


    # Define the position and style of the FPS text
    text_position = (10, 30)  # Position of the FPS text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1  # Font scale (size)
    font_color = (255, 0, 0)  # Font color (BGR format)
    font_thickness = 2  # Font thickness

    #Loading the model
    print('Loading {} with {} labels , the programme start ....'.format(args.model, args.labels))
    loading_animation()
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()

    #Loading labels
    print('Loading labels ...')
    loading_animation()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)

    #Loading camera
    print('Open camera ...')
    loading_animation()
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 15)


    #Tracker instance
    mot_tracker = Sort(max_age=3,min_hits=4,iou_threshold=0.4) #create instance of the SORT tracker

    while cap.isOpened():

        if button.read()==1:
            ret, frame = cap.read()
            if not ret:
                print('No data , please check the camera!')
                break

            fps = cap.get(cv2.CAP_PROP_FPS)
            cv2_im = frame

            #Display the people counter
            person_inside = person_counter_enter-person_counter_leave
            cv2.putText(cv2_im, f"Personnes enters: {person_counter_enter}", (10, 80), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
            cv2.putText(cv2_im, f"Personnes inside: {person_inside}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
            cv2.putText(cv2_im, f"Personnes leaves: {person_counter_leave}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)

            #Resize corectly and run interference
            if cv2_im.shape[:2] != inference_size:
                cv2_im = cv2.resize(cv2_im,(800,600))
                cv2_im_RGB = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
                cv2_im_RGB = cv2.resize(cv2_im_RGB, inference_size)
                run_inference(interpreter, cv2_im_RGB.tobytes())
            else:
                run_inference(interpreter, cv2_im.tobytes())

            #Create list of objects
            objs = get_objects(interpreter, args.threshold)[:args.top_k]
    
            #Filtering
            objs = [obj for obj in objs if obj.score >= 0.3 and labels.get(obj.id, obj.id) == "person"]  # Filter objects with score >= 0.3 and label == "person"
            objs.sort(key=lambda obj: obj.score, reverse=True)  # Sort objects by score in descending order


            #Initialise the sort tracker
            detections = np.array([[obj.bbox.xmin, obj.bbox.ymin, obj.bbox.xmax, obj.bbox.ymax, obj.score] for obj in objs])
            trdata = []
            trackerFlag = False

            #Extract dimension of input image
            height, width, _ = cv2_im.shape

            # Draw detection lines
            cv2.line(cv2_im, (line1_x, 0), (line1_x, height), (0, 255, 255), 1)  # Yellow line for the first detection line
            cv2.line(cv2_im, (line2_x, 0), (line2_x, height), (255, 255, 0), 1)  # Cyan line for the second detection line

            #Update tracker and display detection
            if detections.any():
                if mot_tracker != None:
                    trdata = mot_tracker.update(detections)
                    trackerFlag = True
            if len(objs) != 0:
                cv2_im,person_counter_enter,person_counter_leave = append_objs_to_img(cv2_im, inference_size, objs, labels,trdata,trackerFlag,line1_x,line2_x,entered_left_ids,exited_left_ids,height,width,person_counter_enter,entered_right_ids,exited_right_ids,person_counter_leave)

            #Show FPS       
            cv2.putText(cv2_im, f"FPS: {round(fps, 2)}", text_position, font, font_scale, font_color, font_thickness)
            cv2.imshow('frame', cv2_im)
            current_time = datetime.datetime.now()

            #Send data by UART 2 times by day
            if current_time.second == 15: #and #last_send_time.date() != current_time.data()
                send_data(person_counter_enter)
                send_data(person_counter_leave)
                last_send_time = current_time

            if current_time.second == 45: #and last_send_time.date() != current_time.date():
                send_data(person_counter_enter)
                send_data(person_counter_leave)
                last_send_time = current_time

            #Stop the detection
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        else:
            print('waiting')
            time.sleep(1)

    button.close()
    cap.release()
    cv2.destroyAllWindows()
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Aniamtion ^^
def loading_animation():
    animation = "|/-\\"
    for i in range(10):
        time.sleep(0.1)  # Pause pour donner l'effet d'animation
        print(f"\rLoading {animation[i % len(animation)]}", end="")

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#use to send data
def send_data(data):

    # Convert int to bytes
    bytes_data = struct.pack('i', data)

    # Send data
    port.write(bytes_data)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Use to draw the bbox
def append_objs_to_img(cv2_im, inference_size, objs, labels,trdata,trackerFlag,line1_x,line2_x,entered_left_ids,exited_left_ids,height,width,person_counter_enter,entered_right_ids,exited_right_ids,person_counter_leave):

    #Define scale between inference size and input size
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]

    #Extract informations about bbox of detection
    if trackerFlag and (np.array(trdata)).size:
        for td in trdata:
            x0, y0, x1, y1, trackID = td[0].item(), td[1].item(), td[2].item(), td[3].item(), td[4].item()
            dx0 = [int(ob.bbox.xmin) for ob in objs]
            dy0 = [int(ob.bbox.ymin) for ob in objs]
            dx1 = [int(ob.bbox.xmax) for ob in objs]
            dy1 = [int(ob.bbox.ymax) for ob in objs]
            overlap = np.maximum(0, (np.minimum(dx1, x1) - np.maximum(dx0, x0)) * (np.minimum(dy1, y1) - np.maximum(dy0, y0)))
            max_overlap_index = np.argmax(overlap)
            obj = objs[max_overlap_index]

            # Scale to source coordinate space.
            x, y, w, h = int(x0 * scale_x),int( y0 * scale_y),int( x1 * scale_x), int(y1 * scale_y)
            cx = (x + w) // 2

            #Labels
            percent = int(100 * obj.score)
            label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

            # Draw the bbox
            cv2.rectangle(cv2_im, (x, y), (w, h), (0, 0, 0), 1)  # black (BGR: (0, 0, 0))
            cv2.putText(cv2_im, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)  # green (BGR: (0, 255, 0))
            cv2.putText(cv2_im, str(trackID), (x, y + 60), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 1)  #  red (BGR: (0, 0, 255))

            # Check if the object crosses the detection lines

            #Left
            if line1_x < cx < 300 and trackID not in entered_left_ids and trackID not in entered_right_ids:
                entered_left_ids.add(trackID)

            if line2_x < cx and trackID in entered_left_ids and trackID not in exited_left_ids:
                exited_left_ids.add(trackID)
                person_counter_enter = person_counter_enter + 1

            #Right
            if 550 < cx < line2_x and trackID not in entered_right_ids and trackID not in entered_left_ids:
                entered_right_ids.add(trackID)

            if cx < line1_x and trackID in entered_right_ids and trackID not in exited_right_ids:
                exited_right_ids.add(trackID)
                person_counter_leave =person_counter_leave +1

    else:
        for obj in objs:
            bbox = obj.bbox.scale(scale_x, scale_y)
            x, y = int(bbox.xmin), int(bbox.ymin)
            w, h = int(bbox.xmax), int(bbox.ymax)

            #Labels
            percent = int(100 * obj.score)
            label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

            #Draw the bbox
            cv2_im = cv2.rectangle(cv2_im, (x, y), (w, h), (0, 255, 0), 2)
            cv2_im = cv2.putText(cv2_im, label, (x, y+30),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im,person_counter_enter,person_counter_leave

#--------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Begining
if __name__ == '__main__':
    #Allow serial port usage
    command = "sudo chmod a+rw /dev/ttyS1"
    subprocess.run(command, shell=True)

    #Define motion sensor
    button = GPIO("/dev/gpiochip0", 13, "in")  # pin 36
    port = serial.Serial("/dev/ttyS1", baudrate=9600, timeout =1)
    main()
