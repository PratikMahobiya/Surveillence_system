import tensorflow as tf
import csv
import cv2
import time
import tinys3
import os
import glob
import numpy as np
from utils import visualization_utils as vis_util
from twilio.rest import Client

lastchk_CALL = time.perf_counter()
lastchk_CONN = time.perf_counter()
lastchk_DEL = time.perf_counter()

def upload_data(conn, img_name):

	# Upload a file TO server
	img_file = open("output/" + img_name + ".jpg",'rb')
	conn.upload('balco_Surveillance/{}.jpg'.format(img_name), img_file, 'aonapps')
	# print("Uploaded")

def calling_sms(img_name):

    account_sid = 'YOUR SID'
    auth_token = 'YOUR TOKEN'
    client = Client(account_sid, auth_token)

    # MOBILE NUMBERS -- ADD MORE NUMBER HERE......
    mobile_number = {"Pratik":'+917000xxxxx'}

    print("--------------------------------------Check Your Whatsapp-----------------------------------------")
    for name, num in mobile_number.items():
        
        # # Whatsapp Message code:
        # message = client.messages \
        #         .create(
        #              body="Here's that picture of detected person.",
        #              media_url=['https://aonapps.s3-ap-southeast-1.amazonaws.com/balco_Surveillance/{}.jpg'.format(img_name)],
        #              from_='whatsapp:+14155238886',
        #              to='whatsapp:{}'.format(num)
        #          )

        # # Make Call:
        # call = client.calls.create(
        #                 url='http://demo.twilio.com/docs/voice.xml',
        #                 to= num,
        #                 from_='+19046377572'
        #             )
        print("Sent to :------------------------------   ", name)
        print("Call to :------------------------------   ", name)


def targeted_object_counting(detection_graph, category_index, is_color_recognition_enabled, targeted_object):

        # input video
        cap = cv2.VideoCapture(0)

        global lastchk_CALL, lastchk_CONN,  lastchk_DEL

        the_result = "..."      
        
        with detection_graph.as_default():
          with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # for all the frames that are extracted from input video
            while(cap.isOpened()):
                ret, frame = cap.read()                

                if not  ret:
                    print("end of the video file...")
                    break
                
                input_frame = frame

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Visualization of the results of a detection.        
                counter, csv_line, the_result = vis_util.visualize_boxes_and_labels_on_image_array(cap.get(1),
                                                                                                      input_frame,
                                                                                                      1,
                                                                                                      is_color_recognition_enabled,
                                                                                                      np.squeeze(boxes),
                                                                                                      np.squeeze(classes).astype(np.int32),
                                                                                                      np.squeeze(scores),
                                                                                                      category_index,
                                                                                                      targeted_objects=targeted_object,
                                                                                                      use_normalized_coordinates=True,
                                                                                                      line_thickness=4)
                if(len(the_result) == 0):
                    cv2.putText(input_frame, "Person:- ...", (10, 35), font, 0.8, (0,150,255),2,cv2.FONT_HERSHEY_SIMPLEX)                       
                
                else:                
                    cv2.putText(input_frame, "Person:- "+the_result[10:], (10, 35), font, 0.8, (0,0,255),2,cv2.FONT_HERSHEY_SIMPLEX)

                    # To call Funtion at 10 min interval
                    check = time.perf_counter()

                    # Capture image and store (System Storage)
                    now = time.localtime()
                    img_name = time.strftime("%Y_%b_%d_%H_%M_%S")# format to save images
                    cv2.imwrite("output/" + img_name + ".jpg", input_frame)

                    # TO CREATE CONNECTION WITH S3------------------------------------------------------------------------------------
                    if check > lastchk_CONN:
                    	S3_ACCESS_KEY = "YOUR AWS S3 ACCESS KEY"
                    	S3_SECRET_KEY = "YOUR AWS S3 SECRET KEY"
                    	conn = tinys3.Connection(S3_ACCESS_KEY,S3_SECRET_KEY, tls=True, endpoint='s3-ap-southeast-1.amazonaws.com')
                    	print("--------------------------------------Connected to SERVER-----------------------------------------")

                    	# Execute this part after (5 Min [300])
                    	lastchk_CONN = check + 	300		#( 10 sec default)

                    # Upload Data to Server
                    upload_data(conn, img_name)

                    # Delete the images from folder
                    if check > lastchk_DEL:
                    	path = "output/*"
                    	folder = glob.glob(path)
                    	for img in folder:
                    		os.remove(img)
                    	print("--------------------------------------------Deleted-----------------------------------------------")

                    	# Execute this after (10 sec)
                    	lastchk_DEL = check + 10		#( 10 sec default)

                    # Make a CALL AND MESSAGE
                    if check > lastchk_CALL:
                    	calling_sms(img_name)

                    	# Execute this after (10 Min [6000])
                    	lastchk_CALL = check + 30		#( 30 sec default)

                cv2.imshow('Surveillance',input_frame)

                # output_movie.write(input_frame)
                # print ("writing frame")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            cap.release()
            cv2.destroyAllWindows()
