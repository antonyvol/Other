import cv2

vs = cv2.VideoCapture(0)

fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
min_hull_area = 2000
hull_cntr = 0
current_hull = None
frames_to_skip = 10

while(True):
	ret, frame = vs.read()

	fgmask = fgbg.apply(frame)
	fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

	_, contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	hull = None
	max_hull_area = 0
	for contour in contours:
		if cv2.contourArea(contour) > max_hull_area and cv2.contourArea(contour) > min_hull_area:
			max_hull_area = cv2.contourArea(contour)
			hull = cv2.convexHull(contour)
	
	if hull is not None:
		if current_hull is None:
			current_hull = hull
			continue

		hull_cntr += 1 
		if hull_cntr % frames_to_skip == 0:
			current_hull = hull
			hull_cntr = 0
	
	if current_hull is not None:
		M = cv2.moments(current_hull)
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
		cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
		cv2.drawContours(frame, [current_hull], 0, (0,255,0), thickness=5)

	cv2.imshow('fgmask', frame)

	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

vs.release()
cv2.destroyAllWindows()