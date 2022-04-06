# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 18:47:31 2022

@author: pooja
"""
import cv2 as cv
import numpy as np
import sys
import random


RANSAC_ITERS = 100
DEFAULT_TEMPLATE_SIZE = 15
VERY_SMALL_VALUE = 1e-10

class StereoCam:
    def __init__(self, img_l, img_r, camera_matrix_l, camera_matrix_r, baseline, focal_length, resize_ratio=1):
        self.scale = resize_ratio
        self.height, self.width,_ = img_l.shape
        self.B = baseline
        self.f = focal_length
        self.K_l = camera_matrix_l
        self.K_r = camera_matrix_r
        
        self.F, pts_l, pts_r = self.findFundamentalMatrix(img_l, img_r)
        self.E = self.findEssentialMatrix()
        self.R_l, self.t_l = self.findPose(self.E, self.K_l)
        self.R_r, self.t_r = self.findPose(self.E, self.K_r)
        self.H_l, self.H_r = self.rectifyUncalibrated(pts_l, pts_r)
    
    def findGoodCorrespondences(self, img_l, img_r, count):
        orb = cv.ORB_create()
        kp_l, des_l = orb.detectAndCompute(img_l, None)
        kp_r, des_r = orb.detectAndCompute(img_r, None)
        
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_l, des_r)
        matches= sorted(matches, key = lambda x:x.distance)
        
        good = []
        pts_l = []
        pts_r = []
        
        for match in matches[0:count]:
        
                good.append(match)
                pts_r.append(kp_r[match.trainIdx].pt)
                pts_l.append(kp_l[match.queryIdx].pt)
        
        
        pts_l = np.int32(pts_l)
        pts_r = np.int32(pts_r)
        
        return pts_l, pts_r
    
    def findFundamentalMatrix(self, img_l, img_r):
        TOP_N_CORRESPONDENCES = 50
        pts_l, pts_r = self.findGoodCorrespondences(img_l, img_r, TOP_N_CORRESPONDENCES)
        
        min_zero = sys.maxsize
        NO_OF_POINTS = 8                                                       # 8-point algorithm

        A=[]
        for iter in range(RANSAC_ITERS):
        
            indx = random.sample(range(len(pts_l)), NO_OF_POINTS)
            # indx = np.random.choice(len(pts0), 8, False)
            
            set_l = np.array(pts_l)[indx]
            set_r = np.array(pts_r)[indx]
            # print(set0)
            # print(set1)
            for i in range(NO_OF_POINTS):
                x, y = set_l[i]
                x_dash, y_dash = set_r[i]
                new_row = [ x_dash*x, x_dash*y, x_dash, y_dash*x, y_dash*y, y_dash, x, y, 1]
                A.append(new_row)
            
            U, S, VT = np.linalg.svd(A)
            l = VT[-1,:]/VT[-1,-1]
            F_iter = l.reshape(3,3)
            
        
            p_l = np.append(pts_l[0],1)
            p_r = np.append(pts_r[0],1)
            
            new_min_zero = np.matmul(np.transpose(p_r),np.matmul(F_iter,p_l))
            
        
            if abs(new_min_zero)< min_zero:
                min_zero = new_min_zero
                F = F_iter
        F_iter= np.array([[2.45669299*pow(10,-6), -7.66313507*pow(10,-5),  3.46524708*pow(10,-2)],
                  [ 7.82618730*pow(10,-5),  5.01377959*pow(10,-6), -1.35213400*pow(10,10)],
                  [-3.62422345*pow(10,-2),  1.35213400*pow(10,10),  1.00000000]])
        return F_iter, pts_l, pts_r
    
    
    def findEssentialMatrix(self):
        E = np.matmul(np.transpose(self.K_r),np.matmul(self.F, self.K_l))
        return E
    
    def findPose(E,K):
        U, S, VT = np.linalg.svd(E)
        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        W_inv = np.linalg.inv(W)
        R = np.matmul(np.matmul(U, W_inv), VT)
        t = np.matmul(np.matmul(np.matmul(U, W), S), U.T)
        return R, t
        
    def rectifyUncalibrated(self, pts_l, pts_r):
        _, H_l, H_r = cv.stereoRectifyUncalibrated(pts_l, pts_r, self.F, imgSize=(self.width, self.height))
        return H_l, H_r
    
    def transformProjectionUncalibrated(self, img_l, img_r):
        img_l_rectified = cv.warpPerspective(img_l, self.H_l, (self.width, self.height))
        img_r_rectified = cv.warpPerspective(img_r, self.H_r, (self.width, self.height))
        
        return img_l_rectified, img_r_rectified
        
    def getDisparityMap(self, img_l, img_r, template_size = DEFAULT_TEMPLATE_SIZE):
        img_l_rectified, img_r_rectified = self.transformProjectionUncalibrated(img_l, img_r)
        
        gray_l = cv.cvtColor(img_l_rectified, cv.COLOR_BGR2GRAY)
        gray_r = cv.cvtColor(img_r_rectified, cv.COLOR_BGR2GRAY)
        
        r,c = self.height, self.width

        pad = int(template_size/2)
        w = int(c/10)
        disp = np.zeros_like(gray_l)
        
        padded_l= cv.copyMakeBorder(gray_l, pad, pad, pad, pad, cv.BORDER_CONSTANT,value= [0,0,0])
        padded_r= cv.copyMakeBorder(gray_r, pad, pad, pad, pad, cv.BORDER_CONSTANT,value= [0,0,0])
        
        for y in range(pad,pad+r):
            for xl in range(pad,pad+c): 
                min_SAD = sys.maxsize
                template_l = np.array(padded_l[y-pad:y+pad,xl-pad:xl+pad])
                
                for xr in range(xl-w,xl+w):
                    if  xr<pad or xr>pad+c-1:
                        continue
                    
                    template_r = np.array(padded_r[y-pad:y+pad,xr-pad:xr+pad])
                    
                    assert template_l.shape == template_r.shape
                    
                    SAD = np.sum(abs(np.subtract(template_l, template_r)))
                    
                    if min_SAD > SAD:
                        min_SAD = SAD
                        xr_best = xr
                        
                disp[y-pad][xl-pad] =  self.scale*(xl - xr_best)
                
        disp = disp.astype(np.float32)
        disp= disp + VERY_SMALL_VALUE

        return disp
        

        
    def getDepthMap(self, img_l, img_r):
        disp = self.getDisparityMap(img_l, img_r)
        depth = self.B * self.f/np.copy(disp)
        MAX_DEPTH = self.B * self.f/ 1
        depth[depth > MAX_DEPTH] = MAX_DEPTH    
        return disp, depth
        
        
        