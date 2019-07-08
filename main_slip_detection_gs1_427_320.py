#!/usr/bin/env python


from sensor_msgs.msg import CompressedImage,ChannelFloat32
from std_msgs.msg import Bool
import numpy as np
import time
from scipy import ndimage
import matplotlib.pyplot as plt
from visualization_msgs.msg import *
import rospy, math, cv2, os, pickle
import std_srvs.srv

class slip_detection_reaction:

    def __init__(self):
        self.kernal = self.make_kernal(11,'circle')
        self.kernal1 = self.make_kernal(6,'rect')
        self.kernal2 = self.make_kernal(11,'rect')
        self.kernal3 = self.make_kernal(25,'circle')
        self.kernal4 = self.make_kernal(5,'circle')
        self.kernal5 = self.make_kernal(5,'rect')
        self.kernal6 = self.make_kernal(25,'circle')
        self.kernal_size = 25
        self.kernal7 = self.make_kernal(self.kernal_size,'circle')
        self.kernal8 = self.make_kernal(2,'rect')
        self.kernal9 = self.make_kernal(2,'rect')
        self.kernal10 = self.make_kernal(45,'circle')
        self.M = np.load('M_gs1_newSensor_427_320.npy') 
        # self.ROImask = np.load('mask_GS2.npy        self.previous_slip = False
        self.previous_u_sum = np.array([0])
        self.previous_v_sum = np.array([0])
        self.static_flag = False
        self.index = 0 
        self.cols, self.rows, self.cha = 320, 427, 3 
        self.x1_last = []
        self.y1_last = []
        self.highbar_top = 130
        self.lowbar_top = 100
        self.highbar_down = 80
        self.lowbar_down = 60
        self.img_counter = 0
        self.thre = 80
        self.scale = 1
        self.con_flag = False 
        self.refresh = False
        self.restart = False
        self.initial_flag = True
        self.length_flag = True
        self.marker_refflag = True
        self.showimage = True 
        self.thre_slip_dis = 4.5
        self.thre_slip_num = 7
        self.slip_indicator = False 
        self.slip_monitor = ChannelFloat32()
        self.tran_matrix_msg = ChannelFloat32()
        self.slip_pub = rospy.Publisher('/raspicam_node1/slip_monitor', ChannelFloat32, queue_size=1)
        #change to your own rostopic name to subsribe gelslim raw images
        self.image_sub = rospy.Subscriber("/raspicam_node1/image/compressed",CompressedImage,self.call_back,queue_size = 1,buff_size=2**24)
        self.s = rospy.Service('raspicam_node1/restart_detector', std_srvs.srv.Empty, self.restart_detector_server)
        self.useSlipDetection = True #True to detect slip, False to detect collision  
        self.collideThre = 1.5
        

    def rgb2gray(self,rgb):
        return np.dot(rgb[...,:3], [0.5, 0.5, 0.])

    def make_kernal(self,n,type):
        if type is 'circle':
            kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n,n))
        else:
            kernal = cv2.getStructuringElement(cv2.MORPH_RECT,(n,n))
        return kernal 


    def calibration_v2(self,img):
        imgw = cv2.warpPerspective(img, self.M, (self.rows, self.cols))
        # im_temp = imgw*np.dstack((self.ROImask,self.ROImask,self.ROImask))
        # imgwc = im_temp[10:,60:572]
        # imgwc = imgw[5:-10,70:-55,:]
        # imgwc = imgw[3:-10,85:-75,:]
        # imgwc = imgw[:,73:-83,:]
        imgwc = imgw[:,50:-50,:]   #change the crop range 
     
        # cv2.imshow('imgwc',imgwc)
        # cv2.waitKey(1)
        return imgwc

    def defect_mask(self, img):
        background = cv2.GaussianBlur(img[:,:,1].astype(np.float32),(25,15),21)
        im_g = ((img[:,:,1].astype(np.int16)- background +90)*1.1+20).astype(np.uint8)
        # print np.min(im_cal),np.max(im_cal),'minmax'
        im_g = np.clip(im_g,0,255)
        mask = (im_g < 150).astype(np.uint8)
        mask = cv2.erode(mask, self.kernal9, iterations=1)
        # cv2.imshow('mask',mask*255)
        # cv2.imshow('mask_image',im_g)
        # cv2.waitKey(0)
        return mask 

    def make_thre_mask(self,im_cal):
        thre_image = np.zeros(im_cal.shape,dtype = np.uint8)
        previous_mask = np.zeros(im_cal.shape,dtype = np.uint8)
        for i in range(10,80,30):
            _, mask = cv2.threshold(im_cal.astype(np.uint8),i, 255, cv2.THRESH_BINARY_INV)
            mask_expand = cv2.dilate(mask, self.kernal10, iterations=1)
            mask_erode = cv2.erode(mask_expand, self.kernal10, iterations=1)
            thre_image += (mask_erode - previous_mask)/255*i 
            previous_mask = mask_erode
            # cv2.imshow('threshold', thre_image)
            # cv2.waitKey(0)
        thre_image += (np.ones(im_cal.shape, dtype = np.uint8) - previous_mask/255)*80 + 10
       
        return thre_image


    def contact_detection(self,im_cal):
        pad = 20
        self.highbar_top = 120
        self.lowbar_top = 90
        self.highbar_down = 70
        self.lowbar_down = 50
        im_canny_top = cv2.Canny(im_cal[:self.rows*2/3,:].astype(np.uint8),self.lowbar_top,self.highbar_top)
        im_canny_down = cv2.Canny(im_cal[self.rows*2/3:,:].astype(np.uint8),self.lowbar_down,self.highbar_down)

        im_canny = np.concatenate((im_canny_top,im_canny_down),axis = 0)

        im_canny[:pad*2,:pad] = 0
        im_canny[-pad*2:,-pad*1:] = 0
        im_canny[-pad*2:,:pad] = 0
        # im_canny[-pad:,:] = 0
        im_canny = im_canny  * self.de_mask
        # im_canny = mask_blue * im_canny  * mask_brightness
        # cv2.imshow('calibrated image', im_cal.astype(np.uint8))
        # cv2.imshow('edge_image',im_canny) 
        # cv2.waitKey(1)
        img_d = cv2.dilate(im_canny, self.kernal1, iterations=1)
        img_e = cv2.erode(img_d, self.kernal1, iterations=1)
        img_ee = cv2.erode(img_e, self.kernal2, iterations=1)
        contact = cv2.dilate(img_ee, self.kernal3, iterations=1).astype(np.uint8)
        return contact

    def creat_mask(self,im_cal):
        thresh1 = (im_cal < self.thre_image).astype(np.uint8)

        temp1 = cv2.dilate(thresh1, self.kernal9, iterations=1)
        temp2 = cv2.erode(temp1, self.kernal8, iterations=1)
        # cv2.imshow('thresh1',temp2)
        # cv2.waitKey(1)
        final_image1 = cv2.dilate(temp2, self.kernal5, iterations=1)
        # final_image2 = cv2.dilate(final_image1, self.kernal9, iterations=1)
        # cv2.imshow('final_image1',final_image1*255)
        # cv2.waitKey(1)
        return (1-final_image1)*255


    def find_dots(self,binary_image):
        # down_image = cv2.resize(binary_image, None, fx=2, fy=2) 
        params = cv2.SimpleBlobDetector_Params()
        # Change thresholds
        params.minThreshold = 1 
        params.maxThreshold = 12 
        params.minDistBetweenBlobs = 9
        params.filterByArea = True 
        params.minArea = 9
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(binary_image.astype(np.uint8))
        # im_to_show = (np.stack((binary_image,)*3, axis=-1)-100)
        # for i in range(len(keypoints)):   
        #     cv2.circle(im_to_show, (int(keypoints[i].pt[0]), int(keypoints[i].pt[1])), 5, (0, 100, 100), -1)
        
        # cv2.imshow('final_image1',im_to_show)
        # cv2.waitKey(1)
        return keypoints
                
    def flow_calculate_in_contact(self,keypoints2):
        x2, y2, u, v, x1_paired, y1_paired, x2_paired, y2_paired = [], [], [], [], [], [], [], []

        for i in range(len(keypoints2)): 
            x2.append(keypoints2[i].pt[0]/self.scale)
            y2.append(keypoints2[i].pt[1]/self.scale)

        x2 = np.array(x2) 
        y2 = np.array(y2)
        x_initial = list(self.x_iniref)
        y_initial = list(self.y_iniref)
        u_ref = list(self.u_addon)
        v_ref = list(self.v_addon)

        for i in range(x2.shape[0]):
            
            distance = list(((np.array(x_initial) - x2[i])**2 + (np.array(y_initial) - y2[i])**2))
            min_index = distance.index(min(distance))
            u_temp = x2[i] - x_initial[min_index] 
            v_temp = y2[i] - y_initial[min_index] 
            shift_length = np.sqrt(u_temp**2+v_temp**2)
            # print 'length',shift_length

            if shift_length < 7:
                # print xy2.shape,min_index,len(distance)
                x1_paired.append(x_initial[min_index]-u_ref[min_index])
                y1_paired.append(y_initial[min_index]-v_ref[min_index])
                x2_paired.append(x2[i])
                y2_paired.append(y2[i])
                u.append(u_temp + u_ref[min_index])
                v.append(v_temp + v_ref[min_index])

                del x_initial[min_index], y_initial[min_index], u_ref[min_index], v_ref[min_index]

                if shift_length > 4: 
                    self.refresh = True 

        return x1_paired,y1_paired,x2_paired,y2_paired,u,v

    def flow_calculate_global(self,keypoints2):
        x2, y2, u, v, x1_paired, y1_paired, x2_paired, y2_paired  = [], [], [], [], [], [], [], []
        x1_return, y1_return, x2_return, y2_return, u_return, v_return = [],[],[],[],[],[]

        for i in range(len(keypoints2)): 
            x2.append(keypoints2[i].pt[0]/self.scale)
            y2.append(keypoints2[i].pt[1]/self.scale)

        x2 = np.array(x2) 
        y2 = np.array(y2)
        x_initial = list(self.x_iniref)
        y_initial = list(self.y_iniref)
        u_ref = list(self.u_addon)
        v_ref = list(self.v_addon)
        

        for i in range(x2.shape[0]):
            distance = list(((np.array(x_initial) - x2[i])**2 + (np.array(y_initial) - y2[i])**2))
            min_index = distance.index(min(distance))  
            u_temp = x2[i] - x_initial[min_index] 
            v_temp = y2[i] - y_initial[min_index] 
            shift_length = np.sqrt(u_temp**2+v_temp**2)
            # print 'length',shift_length

            # print xy2.shape,min_index,len(distance)
            if shift_length < 7:
                x1_paired.append(x_initial[min_index]-u_ref[min_index])
                y1_paired.append(y_initial[min_index]-v_ref[min_index])
                x2_paired.append(x2[i])
                y2_paired.append(y2[i])
                u.append(u_temp + u_ref[min_index])
                v.append(v_temp + v_ref[min_index])
                # sign = self.ROI[y2[i].astype(np.uint16),x2[i].astype(np.uint16)]
                # x1_return.append((x_initial[min_index]-u_ref[min_index])*sign)
                # y1_return.append((y_initial[min_index]-v_ref[min_index])*sign)
                # x2_return.append((x2[i])*sign)
                # y2_return.append((y2[i])*sign)
                # u_return.append((u_temp + u_ref[min_index])*sign)
                # v_return.append((v_temp + v_ref[min_index])*sign)
                del x_initial[min_index], y_initial[min_index], u_ref[min_index], v_ref[min_index]   

            
        # print len(self.x_iniref), len(x2_paired)
        self.x_iniref = list(x2_paired) 
        self.y_iniref = list(y2_paired)
        self.u_addon = list(u)
        self.v_addon = list(v)
        self.refresh = False 

        inbound_check = self.ROI[np.array(y2_paired).astype(np.uint16),np.array(x2_paired).astype(np.uint16)]*np.array(range(len(x2_paired)))

        final_list = list(set(inbound_check)- set([0]))
        x1_return = np.array(x1_paired)[final_list]
        y1_return = np.array(y1_paired)[final_list]
        x2_return = np.array(x2_paired)[final_list]
        y2_return = np.array(y2_paired)[final_list]
        u_return = np.array(u)[final_list]
        v_return = np.array(v)[final_list]

        return x1_return, y1_return, x2_return, y2_return, u_return, v_return
        # return x1_paired,y1_paired,x2_paired,y2_paired,u,v

    def ROI_calculate(self):
        ROI = cv2.dilate(self.contactmask, self.kernal4, iterations=1)/255
        ROI_big = cv2.dilate(self.contactmask, self.kernal6, iterations=1)/255
        return ROI, ROI_big

    def estimate_uv(self,x2,y2,final_list):
        theta = np.arcsin(self.tran_matrix[1,0])
        x1_select = np.array(self.x1)[final_list]
        y1_select = np.array(self.y1)[final_list]
        u_select = self.u_sum[final_list]
        v_select = self.v_sum[final_list]

        u_mean = np.mean(u_select)
        v_mean = np.mean(v_select)
        x_mean = np.mean(x1_select)
        y_mean = np.mean(y1_select)

        u_estimate = u_mean + theta*(y_mean - np.array(y2))
        v_estimate = v_mean + theta*(np.array(x2)-x_mean)

        # u_estimate[self.trash_list] = 0
        # v_estimate[self.trash_list] = 0
        return u_estimate, v_estimate

    def dispOpticalFlow(self,im_cal,x,y):
        # mask = np.zeros_like(im_cal)
        mask2 = np.zeros_like(im_cal)
        amf = 1
        x = np.array(x).astype(np.int16)
        y = np.array(y).astype(np.int16)
        for i in range(self.u_sum.shape[0]):
             # mask = cv2.line(mask, (int(x[i]-self.acc_u[i]*2),int(y[i]-self.acc_v[i]*2)),(int(x[i]),int(y[i])), [0, 80, 0], 2)
             # mask2 = cv2.line(mask2, (int(x[i]+self.u_diff[i]*3),int(y[i]+self.v_diff[i]*3)),(int(x[i]),int(y[i])), [0, 0, 100], 2)
             if self.useSlipDetection:
                 mask2 = cv2.line(mask2, (int(x[i]+self.u_estimate[i]*amf),int(y[i]+self.v_estimate[i])),(x[i],y[i]), [0, 0, 120], 2)
             mask2 = cv2.line(mask2, (int(x[i]+self.u_sum[i]*amf),int(y[i]+self.v_sum[i]*amf)),(x[i],y[i]), [0, 120, 120], 2)

        img = cv2.add(im_cal/2,mask2)

        img[:,:,1] = img[:,:,1] + self.contactmask/7
        if self.useSlipDetection:
            img[:,:,2] = img[:,:,2] + self.p_region*30

        if self.slip_indicator:
            img = img + self.im_slipsign/2
    
        cv2.imshow("force_flow_1",img.astype(np.uint8))
        cv2.waitKey(1)

        # raw_input("Press Enter to continue...")

    def restart_detector_server(self,req):
        self.restart = True
        print "restart the slip detector" 
        return std_srvs.srv.EmptyResponse()

    def call_back(self,data):
        t = time.time()
        np_arr = np.fromstring(data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        raw_imag = cv2.flip(image_np, 0) 
        self.img_counter += 1
        # print raw_imag.shape 
        # cv2.imwrite('sample_img_gelsight1_small.png', raw_imag.astype(np.uint8))
        # cv2.imshow('raw_image',raw_imag.astype(np.uint8))
        # cv2.waitKey(1)

        if not self.con_flag:

            if self.initial_flag:
                imgwc = self.calibration_v2(raw_imag)
                self.de_mask = self.defect_mask(imgwc)
                # self.img_blur = cv2.GaussianBlur(imgwc.astype(np.float32),(31,31),30)
                self.img_blur = cv2.GaussianBlur(imgwc.astype(np.float32),(25,15),21)
                self.initial_flag = False
                imgwc = self.calibration_v2(raw_imag).astype(np.int16)
                im_gray = self.rgb2gray(imgwc)#.astype(np.uint8)
                ill_back = cv2.GaussianBlur(im_gray,(31,31),31)
                # im_cal = im_gray - ill_back +150
                im_cal = (im_gray - ill_back +50)*2+20
                # print np.min(im_cal),np.max(im_cal),'minmax'
                im_cal = np.clip(im_cal,0,255)
                self.thre_image = self.make_thre_mask(im_cal)


            imgwc = self.calibration_v2(raw_imag).astype(np.int16)
            im_gray = self.rgb2gray(imgwc)#.astype(np.uint8)
            ill_back = cv2.GaussianBlur(im_gray,(31,31),31)
            # im_cal = im_gray - ill_back +150
            im_cal = (im_gray - ill_back +50)*2+20
            # print np.min(im_cal),np.max(im_cal),'minmax'
            im_cal = np.clip(im_cal,0,255)

            # cv2.imshow('im_cal',im_cal.astype(np.uint8))
            # cv2.waitKey(1)

            self.im_slipsign = np.zeros(imgwc.shape)
            self.im_noslipsign = np.zeros(imgwc.shape)
            cv2.putText(self.im_slipsign, 'Slip', (210,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            # cv2.putText(self.im_noslipsign, 'No Slip', (150,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            self.contactmask = self.contact_detection(im_cal)   
            # print np.sum(self.contactmask)
            # print np.sum(self.contactmask)/255
            # final_image = self.creat_mask(imgwc)

            if np.sum(self.contactmask)/255 > 640*480/100: #if there is large contact
                self.index += 1
                if self.index > 3:
                    self.ROI, self.ROI_big =  self.ROI_calculate()

                    # self.x_minb = int(np.amin(self.xv*self.ROI_big+(1-self.ROI_big)*255))
                    # self.x_maxb = int(np.amax(self.xv*self.ROI_big))
                    # self.y_minb = int(np.amin(self.yv*self.ROI_big+(1-self.ROI_big)*255))
                    # self.y_maxb = int(np.amax(self.yv*self.ROI_big))   

                    final_image = self.creat_mask(im_cal)
                    keypoints = self.find_dots(final_image)
                    # im_with_keypoints = cv2.drawKeypoints(final_image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                    self.u_sum = np.zeros(len(keypoints))
                    self.v_sum = np.zeros(len(keypoints))
                    self.u_addon = list(self.u_sum)
                    self.v_addon = list(self.v_sum)
                    self.x1_last = []
                    self.y1_last = []
                    for i in range(len(keypoints)):
                        self.x1_last.append(keypoints[i].pt[0]/self.scale)
                        self.y1_last.append(keypoints[i].pt[1]/self.scale)
                    self.x_initial = list(self.x1_last)
                    self.y_initial = list(self.y1_last)
                    self.x_iniref = list(self.x1_last)
                    self.y_iniref = list(self.y1_last)
                    self.con_flag = True
                    self.index = 0
                    self.previous_image = np.array(final_image)
                    print "finish pre-calculation"
            else:
                print "No contact"
                self.slip_monitor.name = 'False'
                self.slip_monitor.values = [0.,0.,0.]
                self.slip_pub.publish(self.slip_monitor)


        else:  #start detecting slip 
            imgwc = self.calibration_v2(raw_imag).astype(np.float32)
            im_gray = self.rgb2gray(imgwc)#.astype(np.uint8)
            ill_back = cv2.GaussianBlur(im_gray,(31,31),31)
            im_cal = (im_gray - ill_back +50)*2+20
            # print np.min(im_cal),np.max(im_cal),'minmax'
            im_cal = np.clip(im_cal,0,255)
            # cv2.imshow('im_cal',im_cal.astype(np.uint8))
            # cv2.waitKey(1)
            # im_cal = imgwc/self.img_blur*100 
            # self.img_blur = cv2.GaussianBlur(imgwc.astype(np.float32),(51,31),41)
            im_cal_show = (imgwc-self.img_blur)+150
            # cv2.imshow('contact_image',im_cal.astype(np.uint8))
            # cv2.waitKey(1)
            self.contactmask = self.contact_detection(im_cal)   
            self.ROI, self.ROI_big =  self.ROI_calculate()
            if np.sum(self.contactmask)/255 < 600 or self.restart:
                self.con_flag = False
                self.restart = False 
                self.kernal_size = 25
                self.kernal7 = self.make_kernal(self.kernal_size,'circle')
            else: 
                self.p_region = cv2.erode(self.contactmask/255 , self.kernal7, iterations=1) 
                # cv2.imshow('contact_image',(self.contactmask).astype(np.uint8))
                # cv2.waitKey(1)

                final_image = self.creat_mask(im_cal)
                # final_image2 = self.creat_mask_2(im_gray,ill_back)
                if self.refresh:
                    keypoints = self.find_dots(final_image)
                    self.x1,self.y1,x2,y2,u,v = self.flow_calculate_global(keypoints)
                    # print 'reference changed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
                else:
                    keypoints = self.find_dots(final_image*self.ROI)
                    self.x1,self.y1,x2,y2,u,v = self.flow_calculate_in_contact(keypoints)


                # im_with_keypoints = cv2.drawKeypoints(final_image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                # cv2.imshow("marker",im_with_keypoints.astype(np.uint8))
                # cv2.waitKey(1)

                # self.trash_list = sorted(set(self.trash_list))
                self.u_sum = np.array(u)
                self.v_sum = np.array(v)


                if self.useSlipDetection:
          
                    inbound_check = self.p_region[np.array(y2).astype(np.uint16),np.array(x2).astype(np.uint16)]*np.array(range(len(x2)))
                    final_list = list(set(inbound_check)- set([0]))# - set(range(len(u),len(x2))))
                    # print "number of points inside", len(set(inbound_check)) #, np.sum(p_region[np.array(y2).astype(np.uint16),np.array(x2).astype(np.uint16)]), len(x2)
                    if len(final_list) < 4:
                        self.kernal_size -= 10
                        self.kernal_size = np.max((self.kernal_size,1))
                        self.kernal7 = self.make_kernal(self.kernal_size,'circle')

                    # print "number of reference points",len(final_list),len(x2),len(self.trash_list)
                    x2_center = np.expand_dims(np.array(x2)[final_list],axis = 1)
                    y2_center = np.expand_dims(np.array(y2)[final_list],axis = 1)
                    x1_center = np.expand_dims(np.array(self.x1)[final_list],axis = 1)
                    y1_center = np.expand_dims(np.array(self.y1)[final_list],axis = 1)
                    p2_center = np.expand_dims(np.concatenate((x2_center,y2_center),axis = 1),axis = 0)
                    p1_center = np.expand_dims(np.concatenate((x1_center,y1_center),axis = 1),axis = 0)
                    # print len(final_list)
                    if len(final_list)>1:
                        self.tran_matrix = cv2.estimateRigidTransform(p1_center,p2_center,False)
                    

                    # print self.tran_matrix

                        if self.tran_matrix is not None:
                            self.u_estimate, self.v_estimate = self.estimate_uv(x2,y2,final_list)
                            self.vel_diff = np.sqrt((self.u_estimate - self.u_sum)**2 + (self.v_estimate - self.v_sum)**2)
                            self.u_diff = self.u_estimate - self.u_sum
                            self.v_diff = self.v_estimate - self.v_sum
                        if np.abs(np.mean(self.v_sum)) > np.abs(np.mean(self.u_sum)) + 2:
                            self.thre_slip_dis = 3.5
                        else:
                            self.thre_slip_dis = 4.5

                            self.numofslip = np.sum(self.vel_diff > self.thre_slip_dis)
                            # print "number of marker", self.numofslip
            
                            # abs_change_u = np.abs(self.previous_u_sum - np.mean(self.u_sum))
                            # abs_change_v = np.abs(self.previous_v_sum - np.mean(self.v_sum))
                            # abs_change = np.sqrt(abs_change_u**2+abs_change_v**2)
                            # diff_img_sum = np.sum(np.abs(self.previous_image.astype(np.int16) - final_image.astype(np.int16)))


                            # print abs_change_u, abs_change_v, int(self.static_flag)
                            # print diff_img_sum
                            # print 'abs_change_u', abs_change_u, 'abs_change_v', abs_change_v
                            
                            # if (abs_change_u < 0.05 + int(self.static_flag)*0.05 and abs_change_v < 0.05 + int(self.static_flag)*0.05):
                            #     self.slip_indicator = False
                            #     self.static_flag = True
                            # else:
                            self.slip_indicator = self.numofslip > self.thre_slip_num
                            self.static_flag = False
                          
                            # raw_input("Press Enter to continue...")
                            if self.showimage:
                                self.dispOpticalFlow(im_cal_show,x2,y2)

                    else: 
                        self.ROI, self.ROI_big =  self.ROI_calculate()
                else:
                    x2_center = np.expand_dims(np.array(x2),axis = 1)
                    y2_center = np.expand_dims(np.array(y2),axis = 1)
                    x1_center = np.expand_dims(np.array(self.x1),axis = 1)
                    y1_center = np.expand_dims(np.array(self.y1),axis = 1)
                    p2_center = np.expand_dims(np.concatenate((x2_center,y2_center),axis = 1),axis = 0)
                    p1_center = np.expand_dims(np.concatenate((x1_center,y1_center),axis = 1),axis = 0)
                    self.tran_matrix = cv2.estimateRigidTransform(p1_center,p2_center,False)
                    self.slip_indicator = np.abs(np.mean(self.v_sum)) > self.collideThre
                    if self.showimage:
                        self.dispOpticalFlow(im_cal_show,x2,y2)


                if self.tran_matrix is None:
                    self.slip_monitor.values = [np.mean(np.array(u)),np.mean(np.array(v)),np.arcsin(self.tran_matrix[1,0])/np.pi*180]
                    if len(final_list) >3:
                        self.slip_indicator = True
                else:
                    self.slip_monitor.values = [np.mean(np.array(u)),np.mean(np.array(v)),0.]

                self.slip_monitor.name = str(self.slip_indicator)
                self.slip_pub.publish(self.slip_monitor)

                self.previous_slip = self.slip_indicator
                self.previous_u_sum = np.mean(np.array(u))
                self.previous_v_sum = np.mean(np.array(v))

                
                if self.slip_indicator: 
                    print("slip!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!") 
                    self.slip_indicator = False
                    self.kernal_size = 25
                    self.kernal7 = self.make_kernal(self.kernal_size,'circle')
                    self.ROI, self.ROI_big =  self.ROI_calculate()

        print 1/(time.time()-t)

def close_gripper_f(grasp_speed=50, grasp_force=15):
    graspinGripper(grasp_speed=grasp_speed, grasp_force=grasp_force)

def open_gripper():
    open(speed=50)
   
def main():
    print "start"
    rospy.init_node('slip_detector', anonymous=True)
    while not rospy.is_shutdown():
        # time.sleep(2)
        # open_gripper()
        # time.sleep(1)
        # force_initial = 10
        # close_gripper_f(50,force_initial)
        # time.sleep(0.1)
        slip_detector = slip_detection_reaction()
        rospy.spin()


if __name__ == "__main__": 
    main()
    
#%%
    
    
