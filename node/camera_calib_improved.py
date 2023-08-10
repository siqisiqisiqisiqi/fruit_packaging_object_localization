#!/usr/bin/env python3

import cv2
import rospy
import torch
import numpy as np
from numpy.linalg import inv
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo


chess_size = 22.86

class CameraCalib:
    def __init__(self):
        rospy.init_node("camera_calib", anonymous=True)

        self.bridge = CvBridge()

        # Init subscribers 
        rospy.Subscriber("zed2i/zed_node/rgb/image_rect_color",
                                         Image, self.get_image)
        # Init subscribers 
        rospy.Subscriber("zed2i/zed_node/depth/depth_registered",
                                         Image, self.get_depth_image)
        camera_info = rospy.wait_for_message("zed2i/zed_node/rgb/camera_info", CameraInfo)
        rospy.sleep(1)
        self.mtx = np.array(camera_info.K).reshape(3,3)
        self.dist = np.array([[0, 0, 0, 0, 0]]).astype("float64")

        self.param_fp = rospy.get_param("~param_fp")
        np.savez(self.param_fp+'/B.npz', mtx=self.mtx, dist=self.dist)

        # Init the yolo model
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5s")

        # Init the publish rate
        self.rvecs = None
        self.tvecs = None
        self.rate = rospy.Rate(10)
        self.predefined_z = -1*35/23/2
        # self.predefined_z = 0
        self.chess_size = 1
        
    def get_depth_image(self, data):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")

        except CvBridgeError as e:
            print(e)

    def get_depth(self, point):
        x = point[0]
        y = point[1]
        depth = 0
        idex = 0
        for i in range(-5,5):
            for j in range(-5,5):
                if np.isnan(self.depth_image[y+j,x+i]):
                    continue
                image_depth = self.depth_image[y+j,x+i][0]
                depth = depth + image_depth
                idex = idex + 1    
        depth = depth/(idex+1e-3)
        self.depth = depth*1000/chess_size

    def get_image(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        # cv2.imshow("Image window", cv_image)
        # cv2.waitKey(3)

    def coordinates_draw(self, img, corners, imgpts):
        corner = tuple(corners[0].astype(int).ravel())
        cv2.line(img, corner, tuple(imgpts[0].astype(int).ravel()), (255,0,0), 5)
        cv2.line(img, corner, tuple(imgpts[1].astype(int).ravel()), (0,255,0), 5)
        cv2.line(img, corner, tuple(imgpts[2].astype(int).ravel()), (0,0,255), 5)
        return img
    
    def drawing(self, img, point, result):
        axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
        origin = np.float32([[0,0,0]]).reshape(-1,3)
        imgpts, _ = cv2.projectPoints(axis, self.rvecs, self.tvecs, self.mtx, self.dist)
        corner, _ = cv2.projectPoints(origin, self.rvecs, self.tvecs, self.mtx, self.dist)
        corner = tuple(corner[0].astype(int).ravel())
        img = cv2.line(img, corner, tuple(imgpts[0].astype(int).ravel()), (255,0,0), 5)
        img = cv2.line(img, corner, tuple(imgpts[1].astype(int).ravel()), (0,255,0), 5)
        img = cv2.line(img, corner, tuple(imgpts[2].astype(int).ravel()), (0,0,255), 5)
        cv2.putText(img, 'X axis', (1100,800), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2, cv2.LINE_AA)
        cv2.putText(img, 'Y axis', (1280,700), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
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
    
    def improved_projection(self, point, mtx, Mat, tvec):
        point2 = inv(mtx)@point
        M = np.eye(3)
        M[:,[2]] = -1*point2
        v1 = np.array([[0.0,0.0,-1*self.depth]]).T
        # rospy.loginfo(f"M matrix shape is {M.shape}")
        result1 = inv(M)@v1
        # rospy.loginfo(f"the result1 value is {result1}, shape is {result1.shape}")
        xc = result1[0,0]
        yc = result1[1,0]
        zc = self.depth
        v2 = np.array([[xc],[yc],[zc]])
        result = inv(Mat)@(v2-tvec)
        # rospy.loginfo(f"final result is {result}")
        return result
    
    def extrinsic_calibration(self):
        img = self.cv_image
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
        axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
        if ret == True:
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

            # Find the rotation and translation vectors.
            ret, self.rvecs, self.tvecs = cv2.solvePnP(objp, corners2, self.mtx, self.dist)

            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(axis, self.rvecs, self.tvecs, self.mtx, self.dist)

            # img = self.coordinates_draw(img,corners2,imgpts)
            # cv2.imshow('img',img)
            # cv2.waitKey(0)
    
    def run(self):

        rospy.sleep(1)
        self.extrinsic_calibration()
        Mat, _ = cv2.Rodrigues(self.rvecs)
        tvec = self.tvecs * self.chess_size

        rospy.sleep(1)
        rospy.loginfo(f"finish the extrinsic calibration !!!!")
        while not rospy.is_shutdown():
            obj = None
            img = np.copy(self.cv_image)
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
                self.get_depth(point)
                result = self.improved_projection(point, self.mtx, Mat, tvec)
                # result = self.projection(point, self.mtx, Mat, tvec)
                rospy.loginfo(f"depth is {self.depth*22.86}")
                result = np.round(result, 2)
                # rospy.loginfo(f"result is {result}")
                # rospy.loginfo(f"the x value is {result[0]}")
                # rospy.loginfo(f"the y value is {result[1]}")
                # rospy.loginfo(f"the z value is {result[2]}")
            except:
                result = []
                point = []

            self.drawing(img, point, result)
            self.rate.sleep()

    
if __name__ == "__main__":
    try:
        node = CameraCalib()
        node.run()
    except rospy.ROSInterruptException:
        pass
    
    cv2.destroyAllWindows()
    rospy.loginfo("Exiting camera calibration node!")