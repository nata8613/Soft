import cv2
import numpy as np
from matplotlib import pyplot as plt
from vector import pnt2line 
from skimage.measure import label  
from skimage.measure import regionprops
from skimage.filters import threshold_adaptive
from skimage.morphology import square, diamond, disk
from skimage.morphology import dilation, erosion, closing, opening
from skimage.filters import threshold_otsu
#za klasifikator
#http://hanzratech.in/2015/02/24/handwritten-digit-recognition-using-opencv-sklearn-and-python.html
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC

#obucavanje klasifikatora
dataset = datasets.fetch_mldata("MNIST Original")
features = np.array(dataset.data, 'int16')
labels = np.array(dataset.target, 'int')
list_hog_fd = []

for feature in features:
	fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
	list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

clf = LinearSVC()
clf.fit(hog_features, labels)



for v in range (0, 10):
    
	#koordinate linije
	imgcoords = [0,0,0,0]
	suma = 0
	j = 0
	#slika sa crnom pozadinom 
	blank_image = np.zeros((480,640,3), np.uint8)

	#brojevi koji su presli liniju
	brojevi = []
	frejmovi = []

	#broj frejma
	frame_no = 0
	korak = 0
	k = 0
	m = 0

	#ucitavanje videa
	cap = cv2.VideoCapture("videos/video-" + str(v) + ".avi")
	#preuzimanje prvog frejma
	ret, img = cap.read()

	#konvertovanje u crno belu sliku i primena morfoloskih operacija
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_TRUNC)
	img_tr_open = opening(thresh, selem=square(4))

	#trazenje svih kontura na prvom frejmu
	im2, contours, hij = cv2.findContours(img_tr_open,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

	for i in range(0, len(contours)-1):
		cnt = contours[i]
		area = cv2.contourArea(cnt)
		
		if area > 1000:
			#upisivanje koordinata najvece konture(linije)
			x,y,w,h = cv2.boundingRect(cnt)
			imgcoords = [x,y,w,h]
			#nagib linije
			k = float(y-y+h)/float(x-x+w)
			m = float(y) - float(x+w)*float(y-y+h)/float(x-x+w)
			y1 = int(-k*(x))
			korak = - y1 + y + h
			
	cap.set(1,40)

	for f in range(40, 1200):
		ret1, frame1 = cap.read()
		
		#uzimanje dela frejma sa linijom
		blank_image[imgcoords[1]-25:imgcoords[1]+imgcoords[3]+25,imgcoords[0]-25:imgcoords[0]+imgcoords[2]+25] = frame1[imgcoords[1]-25:imgcoords[1]+imgcoords[3]+25,imgcoords[0]-25:imgcoords[0]+imgcoords[2]+25]
		
		gray1 = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
		ret2,thresh4 = cv2.threshold(gray1,30,255,cv2.THRESH_TOZERO)#najprecizniji za brojeve
		thresh4 = opening(thresh4, selem=square(2))
		im3, contours1, hij1 = cv2.findContours(thresh4,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		
		
		for n in range(0, len(contours1)-1):
			cnt1 = contours1[n]
			area1 = cv2.contourArea(cnt1)
			M = cv2.moments(cnt1)
			cx = int(M['m10']/M['m00'])
			cy = int(M['m01']/M['m00'])
			
			if area1 > 35 and area1 <1000:
				funkcija = int(-k*(cx))
				funkcija = funkcija + korak +25
				#ako se nalazi ispod linije 
				if funkcija < cy:
					#
					x1,y1,w1,h1 = cv2.boundingRect(cnt1)
					
					cv2.line(frame1,(imgcoords[0]+10,imgcoords[1]+imgcoords[3]),(imgcoords[0]+imgcoords[2]+10, imgcoords[1]),(255,255,255),2)
					
					#racunanje rastojanja od linije kad prodje kontura
					dist, ner0, ner1 = pnt2line((x1, y1), (imgcoords[0]+10,imgcoords[1]+imgcoords[3]), (imgcoords[0]+imgcoords[2]+10, imgcoords[1]))
					#print dist
					if dist > 4 and dist <8.15:
						
						number_image = np.zeros((28,28,3), np.uint8)
						blank_image1 = np.zeros((480,640,3), np.uint8)
						#cv2.drawContours(blank_image1, contours1, n, (255,255,255), 1)
						cv2.drawContours(blank_image1, [cnt1], 0, (1,1,1), 1)
						p1 = 28-w1
						p2 = 28-h1
						#print hij1
						number_image[p2/2:p2/2+h1, p1/2:p1/2+w1] = frame1[y1:y1+h1, x1:x1+w1]
						img_crop_num = blank_image1[cy-14:cy+14, cx-14:cx+14,0]
						duzinaBrojeva = len(brojevi)
						if duzinaBrojeva == 0:
							brojevi.append(img_crop_num)
							frejmovi.append(frame_no)
							cv2.rectangle(frame1,(x1,y1),(x1+w1,y1+h1),(0,255,0),2)
							#cv2.imwrite("image-" + str(j) + ".jpg", number_image)
							j = j + 1
							number_image = number_image[:,:,0]
							#number_image = number_image.reshape(28,28)
							roi_hog_fd = hog(number_image, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
							nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
							print int(nbr[0])
							suma = suma + nbr[0]
							
						else:
							if(img_crop_num.shape == (28,28)):
								poslednja = len(brojevi)-1
								razlika = img_crop_num - brojevi[poslednja]
								razlika = sum(sum(abs(razlika)))
								if razlika >256:
									brojevi.append(img_crop_num)
									frejmovi.append(frame_no)
									cv2.rectangle(frame1,(x1,y1),(x1+w1,y1+h1),(0,255,0),2)
									#cv2.imwrite("image-" + str(j) + ".jpg", number_image)
									j = j + 1
									number_image = number_image[:,:,0]
									#number_image = number_image.reshape(28,28)
									roi_hog_fd = hog(number_image, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
									nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
									print int(nbr[0])
									suma = suma + nbr[0]
						
						
						
		frame_no = frame_no + 1

		cv2.imshow('frame',frame1)
		plt.show()
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	
	with open("out.txt", "r") as file:
		data = file.readlines()
		cols = data[v+2].split('\t')
		data[v+2] = cols[0] + '\t' + str(suma) + '\n'
		with open("out.txt", "w") as file:
			file.writelines(data)
	cap.release()
	cv2.destroyAllWindows()