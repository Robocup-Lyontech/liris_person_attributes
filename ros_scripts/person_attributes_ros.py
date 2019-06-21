#!/usr/bin/python3.5

from person_attributes import person_inference

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

#import cv2

import sys
#print(sys.version) #DBG

# change path order to fix ROS-Kinectic issue
#print(sys.path) #DBG
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
#print(sys.path) #DBG

cvBridge = CvBridge()
frameCnt = 0

dataset_name = 'RAPPETA'
model_filename = 'model_190519-100943_epoch-09_batch-10080'
doGPU = True
person_attributes = person_inference.Person_attributes(dataset_name, model_filename, doGPU)

#client_window_name = "image_client"
#client_window = cv2.namedWindow(client_window_name, cv2.WINDOW_NORMAL)
#cv2.resizeWindow(client_window, 200, 200)
#gesture_window_name = "gesture"
#gesture_window = cv2.namedWindow(gesture_window_name, cv2.WINDOW_NORMAL)
#cv2.resizeWindow(client_window, 200, 200)
#font = cv2.FONT_HERSHEY_DUPLEX


def print_to_frame(frame, text):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 255), cv2.cv.CV_FILLED)
    cv2.putText(frame, text, (10, 50), font, 2, (255, 255, 255), 2)


def callback(data):
    global frameCnt

    #frame = cvBridge.imgmsg_to_cv2(data, 'bgr8')
    frame = cvBridge.imgmsg_to_cv2(data, 'rgb8')
    #rospy.loginfo(rospy.get_caller_id() + ' Image received')

    frameCnt += 1
    comment = 'frame #' + str(frameCnt)
    print(comment)
    #print_to_frame(frame, comment)
    #cv2.imshow(client_window_name, frame)
    #cv2.waitKey(3)

    prediction = person_attributes.predict(frame)
    for p in prediction:
      label, pred, confidence = p
      #print(label + ' ' + str(pred))
      print('{:23s} {:d}    {:.2f}'.format(label, pred, confidence))
    print()

    #if gesture:
    #    print 'gesture=', gesture
    #    prob = float(gesture.split('%')[0])
    #    if prob >= 70:
    #        print_to_frame(frame8, gesture)
    #        cv2.imshow(gesture_window_name, frame8)
    #        cv2.waitKey(3)


# for use inside ROS
if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('videofile/image_raw', Image, callback, queue_size=1, buff_size=10*1024*1024)
    # queue_size=1 is not enough to drop extra frames,
    # buff_size MUST be set to a proper value too
    # see https://answers.ros.org/question/220502/image-subscriber-lag-despite-queue-1
    print('Waiting for images...')
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
    
