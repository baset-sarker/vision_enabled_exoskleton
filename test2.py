import time
from ST_VL6180X import VL6180X
from ADXL345 import adxl_default,getAxes

#Constants
#sensor0_i2cid = 0x29
#sensor0 = VL6180X(sensor0_i2cid)
#sensor0.get_identification()

#if sensor0.idModel != 0xB4:
#    print("Not Valid Sensor, Id reported as ",hex(sensor0.idModel))
#else:
#    print("Valid Sensor, ID reported as ",hex(sensor0.idModel))


#Initialize and report Sensor 0
sensor0_i2cid = 0x29
sensor0 = VL6180X(sensor0_i2cid)
sensor0.get_identification()
if sensor0.idModel != 0xB4:
    print("Not Valid Sensor, Id reported as ",hex(sensor0.idModel))
else:
    print("Valid Sensor, ID reported as ",hex(sensor0.idModel))
bus = sensor0.default_settings()
adxl_default(bus)

while(True):
    L0 = str(sensor0.get_distance())
    print("distance",L0)
    getAxes(bus)
    time.sleep(1)

