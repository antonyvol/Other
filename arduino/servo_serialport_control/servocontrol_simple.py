import serial
from time import sleep

arduino = serial.Serial('/dev/ttyACM0', 9600)   

print('Initializing connection...')
# init servo position to 90 - middle
arduino.write("90".encode())
current_servo_pos = 90 
print('Done.')

from imutils.video import VideoStream
import imutils
import cv2
import time
  
def merge_bounding_boxes(cnts):
	x1 = y1 = pow(2, 63) - 1
	x2 = y2 = 0
	cntr = 0
	movement = False

	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < min_area:
			cntr += 1
			continue

		(x, y, w, h) = cv2.boundingRect(c)

		if x < x1:
			x1 = x
		if y < y1:
			y1 = y
		if x + w > x2:
			x2 = x + w
		if y + h > y2:
			y2 = y + h

	if len(cnts) > cntr:
		movement = True

	return movement, x1, x2, y1, y2


min_area = 500
threshhold_boundary = 8
frame_width = 500
delay = 0.08

x_center = frame_width / 2
mod = 8
servo_threshold = 13 # approx 0-25
is_tracking = True

vs = VideoStream(src=2).start()
 
prevFrame = None

cntr = 0
# loop over the frames of the video
while True:
	frame = vs.read()
	text = "No movement"
 
	frame = imutils.resize(frame, width=frame_width)
	frame = cv2.flip(frame, +1)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)
 
	if prevFrame is None:
		prevFrame = gray
		continue

	frameDelta = cv2.absdiff(prevFrame, gray)
	thresh = cv2.threshold(frameDelta, threshhold_boundary, 255, cv2.THRESH_BINARY)[1]

	thresh = cv2.dilate(thresh, None, iterations=2)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	movement, x1, x2, y1, y2 = merge_bounding_boxes(cnts)

	if movement:
		cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
		text = "Movement"

		x_mid = (x2 + x1) / 2
		# shift is negative for values to the left of center, positive to the right of center
		shift = -1 * (x_center - x_mid) / mod # mod differs for distance from camera, minus because flipped frame 

		new_servo_pos = current_servo_pos + shift
		# limit 0 - 180
		new_servo_pos = min(new_servo_pos, 180)
		new_servo_pos = max(0, new_servo_pos)

		#print("x1 {} x2 {} xmid {} xcenter {}".format(x1, x2, x_mid, x_center))
		print("shift {}".format(shift))
		print("curpos {} tempnewpos {}\n".format(current_servo_pos, new_servo_pos))
		
		cntr += 1
		if cntr % 10 == 0:
			is_tracking = True

		if abs(new_servo_pos - current_servo_pos) > servo_threshold and is_tracking:
			is_tracking = False
			print(int(new_servo_pos))
			arduino.write(str(int(new_servo_pos)).encode())
			print("curpos {} newpos {}".format(current_servo_pos, new_servo_pos))
			current_servo_pos = new_servo_pos
			sleep(0.5)

		time.sleep(delay)

	cv2.putText(frame, "Status: {}".format(text), (10, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

	#cv2.imshow('Threshold', thresh)
	#cv2.imshow("Movement detection", frame)
	#cv2.imshow('Blur', gray)

	prevFrame = gray

	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

vs.stop()
cv2.destroyAllWindows()



