# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A demo that runs object detection on camera frames using OpenCV.

TEST_DATA=../all_models

Run face detection model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite

Run coco model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels ${TEST_DATA}/coco_labels.txt

"""

import time

time.sleep(2)

import cv2
from ST_VL6180X import VL6180X
from ADXL345 import adxl_default,getAxes


from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference
from periphery import GPIO


#from periphery import Serial
#serial = Serial("/dev/ttyS1", 9600)    # pins 29/31 (9600 baud)

hand_state = 0 # 1 means open 0 means close , 2 means close with object

pin_13_out_led   = GPIO("/dev/gpiochip0", 38, "out")  # pin 38
motor = GPIO("/dev/gpiochip0", 9, "out")    # pin 11
solenoid = GPIO("/dev/gpiochip0", 36, "out")  # pin 12
pin_13_out_led   = GPIO("/dev/gpiochip0", 10, "out")  # pin 13

to1 = time.time()
to2 = time.time()


#blink led five times on start
for i in range(0,2):
    pin_13_out_led.write(True)
    time.sleep(1)
    pin_13_out_led.write(False)
    time.sleep(1)

def open_hand():
    global hand_state,to1,to2
    solenoid.write(False)
    motor.write(True)
    time.sleep(2.5)
    motor.write(False)
    #serial.write(b"1")
    hand_state = 1        
    print("Hand Open")
    time.sleep(3)
    to1 = time.time()

def close_hand_with_object():
    global hand_state
    motor.write(False)
    solenoid.write(True)
    time.sleep(2)
    solenoid.write(False)
    #serial.write(b"1")  
    hand_state = 2
    print("Command Hand close with object")
    #end simulation

def close_hand():
    global hand_state,to1,to2
    motor.write(False)
    solenoid.write(True)
    time.sleep(1)
    solenoid.write(False)
    #serial.write(b"1")  
    hand_state = 0
    print("Command Hand close")
    to1 = time.time()
    #end simulation

#initialize hand
close_hand()


#Initialize and report Sensor 0
sensor0_i2cid = 0x29
sensor0 = VL6180X(sensor0_i2cid)
sensor0.get_identification()
if sensor0.idModel != 0xB4:
    print("Not Valid Sensor, Id reported as ",hex(sensor0.idModel))
else:
    print("Valid Sensor, ID reported as ",hex(sensor0.idModel))


bus = sensor0.default_settings()
#adxl default setting
adxl_default(bus)




def check_and_relese_object():
    global bus,hand_state
    
    while True:
        x,y,z = getAxes(bus)
        print(x,y,z)    
        if x < 0.0 and z > 10.0:
            open_hand()    
            print("## command Hand Open")
            break

#check_and_relese_object()

def check_object_close_or_not(detection_percent,bbox_ratio):
    global hand_state

    # get distance 
    distance0 = sensor0.get_distance()
    print("Distance: ",distance0)

    if hand_state == 0 and detection_percent > 90 and (distance0 > 90 and distance0 < 150):
        open_hand()
        time.sleep(3)

    #if percent > 90 and bbox_ratio > 25:
    if hand_state == 1 and detection_percent > 85 and bbox_ratio > 35 and (distance0 > 30 and distance0 < 90):
        close_hand_with_object()
        time.sleep(3)

def calculate_framerate(frame_rate_calc,t1,freq):
    #print('FPS: {0:.2f}'.format(frame_rate_calc))
    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1
    return frame_rate_calc


def main():
    global hand_state
    default_model = 'model/efficientdet-lite-pchallange2022_edgetpu.tflite'
    default_labels = 'model/pchallange2022-labels.txt'
    threshold = 0.1 
    top_k = 3 
    
    interpreter = make_interpreter(default_model)
    interpreter.allocate_tensors()
    labels = read_label_file(default_labels)
    inference_size = input_size(interpreter)

    cap = cv2.VideoCapture(0)

    # Initialize frame rate calculation
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()
    
    while cap.isOpened():
         
        # if hand is opened 
        #if hand_state == 1:
        pin_13_out_led.write(True)
        t1 = cv2.getTickCount()
        ret, cv2_im = cap.read()
        if not ret:
            #pin_13_out_led.write(False)
            break
    
        cv2_im = cv2.resize(cv2_im,inference_size)
        cv2_im = cv2.rotate(cv2_im, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        run_inference(interpreter, cv2_im_rgb.tobytes())
        objs = get_objects(interpreter, threshold)[:top_k]

        cv2_im,percent,bbox_ratio = append_objs_to_img(cv2_im, inference_size, objs, labels)

        #distance0 = sensor0.get_distance()
        #print("Distance: ",distance0)

        if hand_state == 2:
            print("Hand is close with object",hand_state)
            check_and_relese_object()

        elif hand_state == 1 or hand_state == 0:
            if hand_state == 1:
                text_to_show = "Open to grab object"

                #handle time out limit for  open hand  10 sec
                to2 = time.time()
                if (to2 - to1) >= 10:
                    text_to_show = f'Time limit exit{to2-to1:.2f}'
                    close_hand()
            else:
                text_to_show = "Initial Close"

            print(text_to_show,hand_state)
            check_object_close_or_not(percent,bbox_ratio)

        
            
        frame_rate_calc = calculate_framerate(frame_rate_calc,t1,freq)
        cv2.imshow('Vision Enable Exoskleton', cv2_im)

        # else:
            # hand is closed so need to check for opening the hand
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
    
    pin_13_out_led.write(False) 
    cap.release()
    cv2.destroyAllWindows()

    #close pin
    pin_13_out_led.close()
    motor.close()
    solenoid.close()
    #serial.close()


def append_objs_to_img(cv2_im, inference_size, objs, labels):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    percent = 0
    bbox_ratio = 0

    for obj in objs:

        #box area  and ratio calculation
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        b_area = (x1-x0) * (y1-y0)
        img_area = height * width

        bbox_ratio = (b_area/img_area)*100
        #print("img_area: ",img_area," box_area: ",b_area," box ratio: ",bbox_ratio)
    

        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))
        
        if percent > 85:
            cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

            return cv2_im,percent,bbox_ratio

    return cv2_im,percent,bbox_ratio



if __name__ == '__main__':
    main()
