#!/usr/bin/env python


from sensor_msgs.msg import CompressedImage,JointState
from std_msgs.msg import Bool
import numpy as np
import time
from scipy import ndimage
import matplotlib.pyplot as plt
from visualization_msgs.msg import *
# from gripper import *
# from ik.helper import *
from visualization_msgs.msg import *
# from robot_comm.srv import *
# from wsg_50_common.msg import Status
import rospy, math, cv2, os, pickle


class slip_detection_reaction:

    def __init__(self):
        self.kernal = self.make_kernal(21)
        self.kernal1 = self.make_kernal(10)
        self.kernal2 = self.make_kernal(18)
        self.kernal3 = self.make_kernal(26)
        self.kernal4 = self.make_kernal(8)
        self.kernal5 = self.make_kernal(3)
        self.kernal6 = self.make_kernal(50) 
        self.kernal_size = 40
        self.kernal7 = self.make_kernal(self.kernal_size)
        self.M = np.load('M_GS2.npy')
        # self.ROImask = np.load('mask_GS2.npy')
        self.index = 0
        self.cols, self.rows, self.cha = 480, 640, 3 
        self.x1_last = []
        self.y1_last = []
        self.trash_list = []
        self.trans_list_last = []
        self.lowbar = 40
        self.highbar = 65
        self.img_counter = 0
        self.xv, self.yv = np.meshgrid(np.linspace(0, 434, 435, endpoint = True),np.linspace(0, 469, 470, endpoint = True))
        self.thre = 50
        self.scale = 1
        self.con_flag = False 
        self.refresh = False
        self.initial_flag = True
        self.length_flag = True
        self.marker_refflag = True
        self.showimage = True 
        self.slip_indicator = False 
        self.pub = rospy.Publisher('slip_monitor', Bool, queue_size=1)
        self.image_sub = rospy.Subscriber("/rpi/gelsight/raw_image2/compressed",CompressedImage,self.call_back)



    #image processing
    def rgb2gray(self,rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

    def make_kernal(self,n):
        kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n,n))
        return kernal 

    def calibration_v2(self,img):
        imgw = cv2.warpPerspective(img, self.M, (self.rows, self.cols))
        # im_temp = imgw*np.dstack((self.ROImask,self.ROImask,self.ROImask))
        # imgwc = im_temp[10:,60:572]
        imgwc = imgw[10:,95:-110,:]
        im_cal = imgwc/self.img_blur*100
        return im_cal,imgwc

    def creat_mask(self,im_cal):
        img_gray = self.rgb2gray(im_cal).astype(np.uint8)
        ret,thresh1 = cv2.threshold(img_gray,self.thre,255,cv2.THRESH_BINARY)
        final_image2 = cv2.erode(thresh1, self.kernal4, iterations=1)
        final_image = cv2.dilate(final_image2, self.kernal5, iterations=1)
        return final_image

    def find_dots(self,binary_image):
        down_image = cv2.resize(binary_image, (0,0), fx=self.scale, fy=self.scale) 
        # cv2.imshow('edge_image',down_image)
        # cv2.waitKey(1)
        params = cv2.SimpleBlobDetector_Params()
        # Change thresholds
        params.minThreshold = 2
        params.maxThreshold = 255
        params.filterByArea = True
        params.minArea = 70*(self.scale)**2
        detector = cv2.SimpleBlobDetector_create()
        keypoints = detector.detect(down_image.astype(np.uint8))
        return keypoints
                
    def flow_calculate_v2(self,keypoints2):
        xy2, u, v, x2, y2 = [], [], [], [], [] 
        for i in range(len(keypoints2)): 
            xy2.append([keypoints2[i].pt[0]/self.scale,keypoints2[i].pt[1]/self.scale])
        xy2 = np.array(xy2) 
        for i in range(len(self.x1_last)):
            distance = list(np.sqrt((self.x1_last[i] - np.array(xy2[:,0]))**2 + (self.y1_last[i] - np.array(xy2[:,1]))**2))
            min_index = distance.index(min(distance))
            u_temp = xy2[min_index,0] - self.x1_last[i]
            v_temp = xy2[min_index,1] - self.y1_last[i]

            if np.sqrt(u_temp**2+v_temp**2) > 12:
                u_temp = 0
                v_temp = 0
                x2.append(self.x1_last[i])
                y2.append(self.y1_last[i])
                self.trash_list.append(i)
            else:
                np.delete(xy2,min_index,0)
                x2.append(xy2[min_index,0])
                y2.append(xy2[min_index,1])

            u.append(u_temp)
            v.append(v_temp)

        # for i in range(len(self.trash_list)):
        #     u[self.trash_list[i]] = 0
        #     v[self.trash_list[i]] = 0
        return x2,y2,u,v


    def flow_calculate_v3(self,keypoints2):
        xy2, u, v, x2, y2 = [], [], [], [], [] 
        self.x2_raw = []
        self.y2_raw = []
        for i in range(len(keypoints2)): 
            xy2.append([keypoints2[i].pt[0]/self.scale,keypoints2[i].pt[1]/self.scale])
            self.x2_raw.append(keypoints2[i].pt[0]/self.scale)
            self.y2_raw.append(keypoints2[i].pt[1]/self.scale)

        xy2 = np.array(xy2) 
        for i in range(len(self.x1_last)):
            if xy2.shape[0] > 0: 
                distance = list(((self.x1_last[i] - np.array(xy2[:,0]))**2 + (self.y1_last[i] - np.array(xy2[:,1]))**2))
                min_index = distance.index(min(distance))
                u_temp = xy2[min_index,0] - self.x1_last[i]
                v_temp = xy2[min_index,1] - self.y1_last[i]

                if np.sqrt(u_temp**2+v_temp**2) > 10:
                    u_temp = 0
                    v_temp = 0
                    x2.append(self.x1_last[i])
                    y2.append(self.y1_last[i])
                    self.trash_list.append(i)
                else:
                    # print xy2.shape,min_index,len(distance)
                    x2.append(xy2[min_index,0])
                    y2.append(xy2[min_index,1])
                    xy2 = np.delete(xy2,min_index,0)
                u.append(u_temp)
                v.append(v_temp)
            else:
                x2.append(self.x1_last[i])
                y2.append(self.y1_last[i])
                u.append(0)
                v.append(0)

        return x2,y2,u,v



    def contact_detection(self,im):
        im_sub = im/self.img_blur*70
        im_gray = self.rgb2gray(im_sub).astype(np.uint8)

        mask_blue = (im[:,:,2]<135).astype(np.uint8)

        # mask_color = ((self.im_tosave[:,:,0]>70).astype(np.uint8))*mask_red
        # cv2.imshow('contact_image',mask_color.astype(np.uint8)*255)
        # cv2.waitKey(0)

        mask_brightness = im_gray < 90.
        # cv2.imshow('contact_image',mask_brightness.astype(np.uint8)*255)
        # cv2.waitKey(0)
        im_canny = cv2.Canny(im_gray,self.lowbar,self.highbar)
        im_canny = mask_blue * im_canny * mask_brightness
        # cv2.imshow('edge_image',im_canny) 
        # cv2.waitKey(0)
        img_d = cv2.dilate(im_canny, self.kernal1, iterations=1)
        img_e = cv2.erode(img_d, self.kernal1, iterations=1)
        img_ee = cv2.erode(img_e, self.kernal2, iterations=1)
        contact = cv2.dilate(img_ee, self.kernal3, iterations=1).astype(np.uint8)
        # contact[:5,:] = 0
        # contact[-5:,:] = 0
        # contact[:,:5] = 0
        # contact[:,-5:] = 0
        return contact

    def ROI_calculate(self):
        ROI = cv2.dilate(self.contactmask, self.kernal4, iterations=1)/255
        ROI_big = cv2.dilate(self.contactmask, self.kernal6, iterations=1)/255
        return ROI, ROI_big

    def estimate_uv(self,x2,y2,final_list):
        theta = np.arcsin(self.tran_matrix[1,0])
        x1_select = np.array(self.x_initial)[final_list]
        y1_select = np.array(self.y_initial)[final_list]
        u_select = self.u_sum[final_list]
        v_select = self.v_sum[final_list]

        u_mean = np.mean(u_select)
        v_mean = np.mean(v_select)
        x_mean = np.mean(x1_select)
        y_mean = np.mean(y1_select)

        u_estimate = u_mean + theta*(y_mean - np.array(y2))
        v_estimate = v_mean + theta*(np.array(x2)-x_mean)

        u_estimate[self.trash_list] = 0
        v_estimate[self.trash_list] = 0
        return u_estimate, v_estimate

    def dispOpticalFlow(self,im_cal,x,y):
        # mask = np.zeros_like(im_cal)
        mask2 = np.zeros_like(im_cal)
        for i in range(self.u_diff.shape[0]):
             # mask = cv2.line(mask, (int(x[i]-self.acc_u[i]*2),int(y[i]-self.acc_v[i]*2)),(int(x[i]),int(y[i])), [0, 80, 0], 2)
             # mask2 = cv2.line(mask2, (int(x[i]+self.u_diff[i]*3),int(y[i]+self.v_diff[i]*3)),(int(x[i]),int(y[i])), [0, 0, 100], 2)
             mask2 = cv2.line(mask2, (int(x[i]+self.u_estimate[i]*4),int(y[i]+self.v_estimate[i]*4)),(int(x[i]),int(y[i])), [0, 0, 80], 2)
             mask2 = cv2.line(mask2, (int(x[i]+self.u_sum[i]*4),int(y[i]+self.v_sum[i]*4)),(int(x[i]),int(y[i])), [0, 80, 80], 2)

        img = cv2.add(im_cal/1.2,mask2)
        # img = im_cal+50
        # img = cv2.add(img,mask)
        # img = cv2.add(img,mask2)
        # edge = np.dstack((np.zeros_like(self.edge_region), self.edge_region*255, np.zeros_like(self.edge_region))).astype(np.float32)
        # edge_ref = np.dstack((np.zeros_like(self.edge_region_ref), self.edge_region_ref*255, np.zeros_like(self.edge_region_ref))).astype(np.float32)
        # img = cv2.add(edge/4,img)
        img[:,:,1] = img[:,:,1] + self.contactmask/7
        img[:,:,2] = img[:,:,2] + self.p_region*30
        if self.slip_indicator:
            img = img + self.im_slipsign/2
        # img = cv2.add(edge_ref/2,img)
        # img = cv2.add(img,self.ROI/2)
        # cv2.imwrite('/home/siyuan/Documents/2019_ICRA_slip_detection/use_this/data/rotation/processed/process_' + str(self.img_counter) + '.jpg',img.astype(np.uint8))
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # for i in range(len(x)):
            # cv2.putText(img,str(self.vel_diff[i]),(int(x[i]),int(y[i])), font, 0.2,(255,255,255),1,cv2.LINE_AA)
        cv2.imshow("force_flow",img.astype(np.uint8))
        cv2.waitKey(1)
        
    def call_back(self,data):
        t = time.time()
        np_arr = np.fromstring(data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        raw_imag = cv2.flip(image_np, 0) 
        self.img_counter += 1
        # cv2.imshow('contact_image',raw_imag.astype(np.uint8))
        # cv2.waitKey(0)

        if not self.con_flag:
            # get the first image and do calibration 
            # self.contactmask = np.ones((self.img_blur.shape[0:2]),dtype=np.uint8)
            # self.ROI = np.ones((self.img_blur.shape[0:2]),dtype=np.uint8)
            # self.contactmask_left = np.ones((self.img_blur.shape[0:2]),dtype=np.uint8)
            # self.contactmask_right = np.ones((self.img_blur.shape[0:2]),dtype=np.uint8)
            # self.edge_region = np.ones((self.img_blur.shape[0:2]),dtype=np.uint8)
            if self.initial_flag:
                self.refimage_cali = cv2.warpPerspective(raw_imag, self.M, (self.rows, self.cols))
                self.ref_finalimage = self.refimage_cali[10:,95:-110,:].astype(np.float32)
                self.img_blur = cv2.GaussianBlur(self.ref_finalimage,(31,31),30)
                # cv2.imshow('contact_image',self.img_blur.astype(np.uint8))
                # cv2.waitKey(0)
                self.initial_flag = False

            im_cal,self.im_tosave = self.calibration_v2(raw_imag)
            # print im_cal.shape
            # cv2.imwrite('/home/siyuan/Documents/2019_ICRA_slip_detection/Incipient_slip_detection_with_GelSlim/data_figure2/warp_'+ str(self.img_counter) + '.jpg',im_tosave)
            self.im_slipsign = np.zeros(im_cal.shape)
            cv2.putText(self.im_slipsign, 'Slip', (300,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            self.contactmask = self.contact_detection(im_cal)   
            # print np.sum(self.contactmask)

            if np.sum(self.contactmask)/255 > 400: #if there is large contact
                self.index += 1
                if self.index > 10:
                    self.ROI, self.ROI_big =  self.ROI_calculate()

                    self.x_minb = int(np.amin(self.xv*self.ROI_big+(1-self.ROI_big)*255))
                    self.x_maxb = int(np.amax(self.xv*self.ROI_big))
                    self.y_minb = int(np.amin(self.yv*self.ROI_big+(1-self.ROI_big)*255))
                    self.y_maxb = int(np.amax(self.yv*self.ROI_big))   

                    final_image = self.creat_mask(im_cal)
                    keypoints = self.find_dots(final_image*self.ROI)
                    # im_with_keypoints = cv2.drawKeypoints(final_image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                    self.u_sum = np.zeros(len(keypoints))
                    self.v_sum = np.zeros(len(keypoints))
                    for i in range(len(keypoints)):
                        self.x1_last.append(keypoints[i].pt[0]/self.scale)
                        self.y1_last.append(keypoints[i].pt[1]/self.scale)
                    self.x_initial = self.x1_last
                    self.y_initial = self.y1_last
                    self.con_flag = True
                    self.index = 0
                    print "finish pre-calculation"
            else:
                print "No contact"
        else:  #start detecting slip 
            im_cal,self.im_tosave = self.calibration_v2(raw_imag)
            self.contactmask = self.contact_detection(im_cal)  
            # self.ROI, self.ROI_big =  self.ROI_calculate()

            self.p_region = cv2.erode(self.contactmask/255 , self.kernal7, iterations=1) 
            # cv2.imshow('contact_image',(p_region*50+self.rgb2gray(im_cal)).astype(np.uint8))
            # cv2.waitKey(1)
            
            # cv2.imwrite('/home/siyuan/Documents/2019_ICRA_slip_detection/Incipient_slip_detection_with_GelSlim/data_figure2/warp_'+ str(self.img_counter) + '.jpg',im_tosave)
            # self.ROI_img = im_cal[self.y_minb:self.y_maxb,self.x_minb:self.x_maxb].astype(np.uint8)

            # np.save('/home/siyuan/Documents/2019_ICRA_slip_detection/use_this/data/rotation/transfer_matrix/matrix_' + str(self.img_counter) + '.npy',self.tran_matrix)
            # print tran_matrix

            final_image = self.creat_mask(im_cal)
            keypoints = self.find_dots(final_image*self.ROI)
            im_with_keypoints = cv2.drawKeypoints(final_image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # cv2.imshow("marker",im_with_keypoints.astype(np.uint8))
            # cv2.waitKey(1)

            x2,y2,u,v = self.flow_calculate_v3(keypoints)

            self.trash_list = sorted(set(self.trash_list))
            self.u_sum += np.array(u)
            self.v_sum += np.array(v)
            self.u_sum[self.trash_list] = 0 
            self.v_sum[self.trash_list] = 0 

            if self.refresh:
                self.x_initial = self.x2_raw
                self.y_initial = self.y2_raw 
                self.x1_last = self.x2_raw
                self.y1_last = self.y2_raw
                x2 = self.x2_raw
                y2 = self.y2_raw
                self.u_sum = np.zeros(len(self.x2_raw))
                self.v_sum = np.zeros(len(self.x2_raw))
                self.trash_list = []
                self.refresh = False 

            # good_list = list(set(range(len(x2)))-set(self.trash_list)) 
            inbound_check = self.p_region[np.array(y2).astype(np.uint16),np.array(x2).astype(np.uint16)]*np.array(range(len(x2)))
            final_list = list(set(inbound_check)- set([0]) - set(self.trash_list))# - set(range(len(u),len(x2))))
            # print "number of points inside", len(set(inbound_check)) #, np.sum(p_region[np.array(y2).astype(np.uint16),np.array(x2).astype(np.uint16)]), len(x2)
            # if len(final_list) < 4:
            #     self.kernal_size -= 10
            #     self.kernal_size = np.max((self.kernal_size,1))
            #     self.kernal7 = self.make_kernal(self.kernal_size)

            # if self.showimage:
            #     if self.marker_refflag:
            #         self.acc_u, self.acc_v = np.asarray(u), np.asarray(v)
            #         self.marker_refflag = False
            #         # self.edge_region = np.zeros_like(im_cal[:,:,0])
            #     else:
            #         index = 0
            #         for i in range(self.acc_u.shape[0]):
            #             if i not in self.trash_list:
            #                 self.acc_u[i] += u[index]
            #                 self.acc_v[i] += v[index]
            #                 index +=1
            #             else:
            #                 self.acc_u[i] = 0
            #                 self.acc_v[i] = 0

            # print "number of reference points",len(final_list),len(x2),len(self.trash_list)
            x2_center = np.expand_dims(np.array(x2)[final_list],axis = 1)
            y2_center = np.expand_dims(np.array(y2)[final_list],axis = 1)
            x1_center = np.expand_dims(np.array(self.x_initial)[final_list],axis = 1)
            y1_center = np.expand_dims(np.array(self.y_initial)[final_list],axis = 1)
            p2_center = np.expand_dims(np.concatenate((x2_center,y2_center),axis = 1),axis = 0)
            p1_center = np.expand_dims(np.concatenate((x1_center,y1_center),axis = 1),axis = 0)
            self.tran_matrix = cv2.estimateRigidTransform(p1_center,p2_center,False)


            if self.tran_matrix is not None:
                self.u_estimate, self.v_estimate = self.estimate_uv(x2,y2,final_list)
                self.vel_diff = np.sqrt((self.u_estimate - self.u_sum)**2 + (self.v_estimate - self.v_sum)**2)
                self.u_diff = self.u_estimate - self.u_sum
                self.v_diff = self.v_estimate - self.v_sum
                self.numofslip = np.sum(self.vel_diff > 4.)
                # print "number of marker", self.numofslip
                self.slip_indicator = self.numofslip > 3
                # raw_input("Press Enter to continue...")
                if self.showimage:
                    self.dispOpticalFlow(im_cal,x2,y2)
            else: 
                self.ROI, self.ROI_big =  self.ROI_calculate()
                self.x_initial = self.x2_raw
                self.y_initial = self.y2_raw 
                self.x1_last = self.x2_raw
                self.y1_last = self.y2_raw
                x2 = self.x2_raw
                y2 = self.y2_raw
                self.u_sum = np.zeros(len(self.x2_raw))
                self.v_sum = np.zeros(len(self.x2_raw))
                self.trash_list = []
                self.refresh = True

            
            if self.tran_matrix is None and len(final_list) >3:
                self.slip_indicator = True


            self.pub.publish(self.slip_indicator)
            
            if self.slip_indicator: 
                print("slip!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!") 
                self.slip_indicator = False
                self.ROI, self.ROI_big =  self.ROI_calculate()
                self.x_initial = self.x2_raw
                self.y_initial = self.y2_raw 
                self.x1_last = self.x2_raw
                self.y1_last = self.y2_raw
                x2 = self.x2_raw
                y2 = self.y2_raw
                self.u_sum = np.zeros(len(self.x2_raw))
                self.v_sum = np.zeros(len(self.x2_raw))
                self.trash_list = []
                self.refresh = True

            




            self.x1_last = x2
            self.y1_last = y2
            # self.trash_list_last = self.trash_list
        # print 1/(time.time()-t)

# def close_gripper_f(grasp_speed=50, grasp_force=15):
#     graspinGripper(grasp_speed=grasp_speed, grasp_force=grasp_force)

# def open_gripper():
#     open(speed=50)

            
def main():
    print "start"
    rospy.init_node('slip_detector', anonymous=True)
    # open_gripper()
    # time.sleep(2)
    # force_initial = 10
    # close_gripper_f(50,force_initial)
    # time.sleep(2)
    slip_detector = slip_detection_reaction()
    rospy.spin()


if __name__ == "__main__": 
    main()
    
#%%
    
    
