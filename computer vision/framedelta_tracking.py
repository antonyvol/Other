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
delay = 0.08

vs = VideoStream(src=0).start()
 
prevFrame = None

# loop over the frames of the video
while True:
	frame = vs.read()
	text = "No movement"
 
	frame = imutils.resize(frame, width=500)
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
		time.sleep(delay)

	cv2.putText(frame, "Status: {}".format(text), (10, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

	cv2.imshow('Threshold', thresh)
	cv2.imshow("Movement detection", frame)
	cv2.imshow('Blur', gray)

	prevFrame = gray

	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

vs.stop()
cv2.destroyAllWindows()



