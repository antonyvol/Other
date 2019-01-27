import cv2

vs = cv2.VideoCapture(0)

fgbg = cv2.bgsegm.createBackgroundSubtractorGMG() # GMG background subtractor
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
min_hull_area = 2000
hull_cntr = 0
current_hull = None
frames_to_skip = 10 # skipping frames 
stable_radius = 200 # radius inside which movement will not be executed

while(True):
	ret, frame = vs.read()

	fgmask = fgbg.apply(frame)
	fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

	_, contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	hull = None
	max_hull_area = 0

	# draw a circle at the center of the frame
	# vs.get(3) = width
	# vs.get(4) = height
	cv2.circle(frame, (int(vs.get(3)/2), int(vs.get(4)/2)), stable_radius, (0, 0, 255), 1)

	# get maximum contour, in other words find movement
	for contour in contours:
		if cv2.contourArea(contour) > max_hull_area and cv2.contourArea(contour) > min_hull_area:
			max_hull_area = cv2.contourArea(contour)
			hull = cv2.convexHull(contour)
	
	# if movement is found
	if hull is not None:
		# init hull if first iteration and hull is None
		if current_hull is None:
			current_hull = hull
			continue
		# skip some frames with movement to avoid twitching
		hull_cntr += 1 
		if hull_cntr % frames_to_skip == 0:
			current_hull = hull
			hull_cntr = 0
	
	# draw centroid
	if current_hull is not None:
		M = cv2.moments(current_hull)
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
		cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
		# cv2.drawContours(frame, [current_hull], 0, (0,255,0), thickness=5) # draw area of movement
		cv2.line(frame, (int(vs.get(3)/2), int(vs.get(4)/2)), (cX, cY), (0, 0, 255), 1)

	cv2.imshow('fgmask', frame)

	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

vs.release()
cv2.destroyAllWindows()