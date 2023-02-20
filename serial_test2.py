import serial

import serial


ser = serial.Serial('/dev/ttyS1',
                    baudrate=9600,
					parity=serial.PARITY_NONE,
					stopbits=serial.STOPBITS_ONE)


ser.write(serial.to_bytes([0x4C,0x12,0x01,0x00,0x03,0x40,0xFB,0x02,0x7a,0x4C,0x4C]))
ser.close()


#values = bytearray([4, 9, 62, 144, 56, 30, 147, 3, 210, 89, 111, 78, 184, 151, 17, 129])
#ser.write(values)


def open_hand():
	pass
def close_hand():
	pass