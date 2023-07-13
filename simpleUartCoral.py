import serial, struct
import os, time

port = serial.Serial("/dev/ttyS1", baudrate=9600, timeout =1)

while True:
    try:
        port.write(struct.pack('hhl', 5, 10, 15))
        time.sleep(5)
    except Exception as e:
        print(e)


        