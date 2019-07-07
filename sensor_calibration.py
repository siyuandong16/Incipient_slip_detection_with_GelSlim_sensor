import cv2 
import numpy as np 
import matplotlib.pyplot as plt 


def calculateM(im,length,width):
    print('select 4 corners of a squared shape.')
    plt.imshow(im.astype(np.uint8))
    coor = plt.ginput(4,60)
    print("clicked", coor)
    x = np.array((coor[0][0],coor[1][0],coor[2][0],coor[3][0]),dtype = np.float32)
    y = np.array((coor[0][1],coor[1][1],coor[2][1],coor[3][1]),dtype = np.float32)
    X = np.zeros((4,),dtype = np.float32);
    X[2] = (x[2]+x[0])/2
    X[0] = X[2]
    X[1] = X[0]+(x[1]-x[0]+x[3]-x[2])/2
    X[3] = X[1]
    diff_y = (x[1]-x[0]+x[3]-x[2])/2/length*width
    Y = np.zeros((4,),dtype = np.float32)  
    Y[0] = y[0] - 25
    Y[2] = Y[0]+diff_y
    Y[1] = Y[0]
    Y[3] = Y[2]
    rect = np.concatenate((np.expand_dims(x,axis = 1), np.expand_dims(y,axis = 1)), axis=1)
    dst = np.concatenate((np.expand_dims(X,axis = 1), np.expand_dims(Y,axis = 1)), axis=1)
    #dst = np.array([[0, 0],[cols - 1, 0],[cols - 1, rows - 1],[0, rows - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return M


if __name__ == "__main__":
    im = cv2.imread('sample_img_gelsight1_427_320.png')  #read a empty raw image 
    calibrate = False  #set to true if do calbiration, False to view the calibrated image
    cali_file_name = 'M_gs1_newSensor_427_320.npy'
    if calibrate:
        M = calculateM(im,14,14)
        np.save(cali_file_name,M)
    else:
        M = np.load(cali_file_name)
    im_warped = cv2.warpPerspective(im, M, (im.shape[1], im.shape[0])) #warped image
    im_warped_cropped = im_warped[:,50:-50,:]
    cv2.imshow('Calibrated image', im_warped_cropped)
    cv2.waitKey(0)