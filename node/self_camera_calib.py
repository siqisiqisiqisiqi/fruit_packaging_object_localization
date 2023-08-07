#!/usr/bin/env python3
import cv2
import rospy
import torch
import numpy as np
from numpy.linalg import inv
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


class CameraCalib:
    def __init__(self):
        rospy.init_node("camera_calib", anonymous=True)

        self.bridge = CvBridge()

        self.image_fp = rospy.get_param("~image_fp")
        raw_caminfo_fp = rospy.get_param("~raw_camera_info")
        rectified_caminfo_fp = rospy.get_param("~rectified_camera_info")
        with np.load(raw_caminfo_fp) as X:
            self.raw_mtx, self.raw_dist, _, _ = [X[i] for i in 
                                                      ('mtx','dist','rvecs','tvecs')]

        with np.load(rectified_caminfo_fp) as X:
            self.mtx, self.dist, _, _ = [X[i] for i in 
                                              ('mtx','dist','rvecs','tvecs')]

        rospy.sleep(2)
        # Init subscribers 
        # rospy.Subscriber("zed2i/zed_node/rgb/image_rect_color",
        #                                  Image, self.get_image)
        rospy.Subscriber("zed2i/zed_node/left_raw/image_raw_color",
                                         Image, self.get_image)
        
        # Init the yolo model
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5s")

        # Init the publish rate
        self.rvecs = None
        self.tvecs = None
        self.rate = rospy.Rate(10)
        self.predefined_z = -1*35/23/2
        self.chess_size = 1
        

    def get_image(self, data):
        try:
            self.raw_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        cv2.imwrite(self.image_fp, self.raw_image)
        self.cv_image = cv2.undistort(self.raw_image, 
                                      self.raw_mtx, self.raw_dist, None, self.raw_mtx)
        # cv2.imshow("Image window", cv_image)
        # cv2.waitKey(3)
    
    def drawing(self, img, point, result):
        axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
        origin = np.float32([[0,0,0]]).reshape(-1,3)
        imgpts, _ = cv2.projectPoints(axis, self.rvecs, self.tvecs, self.mtx, self.dist)
        corner, _ = cv2.projectPoints(origin, self.rvecs, self.tvecs, self.mtx, self.dist)
        corner = tuple(corner[0].astype(int).ravel())
        img = cv2.line(img, corner, tuple(imgpts[0].astype(int).ravel()), (255,0,0), 5)
        img = cv2.line(img, corner, tuple(imgpts[1].astype(int).ravel()), (0,255,0), 5)
        img = cv2.line(img, corner, tuple(imgpts[2].astype(int).ravel()), (0,0,255), 5)
        cv2.putText(img, 'X axis', (1050,950), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2, cv2.LINE_AA)
        cv2.putText(img, 'Y axis', (1280,800), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
        if len(result) != 0:
            cv2.putText(img, f'[{result[0,0]}, {result[1,0]}]', (point[0,0]+50, point[1,0]-50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128,0,128), 2, cv2.LINE_AA)
            cv2.circle(img, (point[0,0], point[1,0]), 5, (128,0,128), 5)
        cv2.imshow('img',img)
        cv2.waitKey(3)

    def projection(self, point, mtx, Mat, tvecs):
        point2 = inv(mtx)@point
        vec_z = Mat[:,[2]]*self.predefined_z
        Mat2 = np.copy(Mat)
        Mat2[:,[2]] = -1*point2
        vec_o = -1*(vec_z + tvecs)
        result = inv(Mat2)@vec_o
        return result
    
    def extrinsic_calibration(self):
        img = self.cv_image
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
        if ret == True:
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

            # Find the rotation and translation vectors.
            ret, self.rvecs, self.tvecs = cv2.solvePnP(objp, corners2, self.mtx, self.dist)
        else:
            rospy.loginfo("Error, can't recognize the corner!!!!!!!")
    
    def run(self):

        rospy.sleep(1)
        self.extrinsic_calibration()
        Mat, _ = cv2.Rodrigues(self.rvecs)
        tvec = self.tvecs[0] * self.chess_size

        rospy.sleep(2)
        while not rospy.is_shutdown():
            obj = None
            img = np.copy(self.cv_image)
            raw_image = np.copy(self.raw_image)
            results = self.model(img)
            data = results.pandas().xyxy[0]
            for i, row in enumerate(data.loc[:, 'class']):
                if row == 64:
                    obj = data.iloc[i]
                    break
            try:
                xmin = int(obj.loc['xmin'])
                xmax = int(obj.loc['xmax'])
                ymin = int(obj.loc['ymin'])
                ymax = int(obj.loc['ymax'])
                xcenter = int((xmin + xmax)/2)
                ycenter = int((ymin + ymax)/2)
                point = np.array([[xcenter,ycenter,1]]).T
                result = self.projection(point, self.mtx, Mat, tvec)
                result = np.round(result, 2)
            except:
                result = []
                point = []

            self.drawing(raw_image, point, result)
            self.rate.sleep()

    
if __name__ == "__main__":
    try:
        node = CameraCalib()
        node.run()
    except rospy.ROSInterruptException:
        pass
    
    cv2.destroyAllWindows()
    rospy.loginfo("Exiting camera calibration node!")