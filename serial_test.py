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


from periphery import Serial
serial = Serial("/dev/ttyS1", 9600)    # pins 29/31 (9600 baud)


#OPEN_HAND,CLOSE_HAND,GRUB_OBJECT
hand_state = 'CLOSE_HAND'
pin_40_control_command = GPIO("/dev/gpiochip0", 39, "out")  # pin 40
pin_38_out_led   = GPIO("/dev/gpiochip0", 38, "out")  # pin 38


#initial state set close hand
pin_40_control_command.write(False)
pin_38_out_led.write(False)
serial.write(b"ABCDEFGHIJKLM") 
#end simulation

# while(True):
#     serial.write(b"12345678901") 
#     print("sent")
#     time.sleep(2)


serial.close()
