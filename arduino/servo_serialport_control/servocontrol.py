import serial
from time import sleep

arduino = serial.Serial('/dev/ttyACM0', 9600)   

print('Initializing connection...')
sleep(1)

while True:                                             
    command = input ("Servo position: ").encode()       
    arduino.write(command)                          
    reachedPos = str(arduino.readline())            
    print(reachedPos)     