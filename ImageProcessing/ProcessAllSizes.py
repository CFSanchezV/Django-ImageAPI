from math import sqrt, pi
import numpy as np
import cv2
from mediapipe.python.solutions import pose as mp_pose

# Utils class to store measurement functions
class MUtils:
    @staticmethod
    def write_image(path, img):
        img = cv2.convertScaleAbs(img, alpha=(255.0))
        cv2.imwrite(path, img)

    @staticmethod
    def convert_image(img):
        img = cv2.convertScaleAbs(img, alpha=(255.0))
        return img

    @staticmethod
    def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image

        # calcs ratios
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)

        else:
            r = width / float(w)
            dim = (width, int(h * r))

        resized = cv2.resize(image, dim, interpolation = inter)
        return resized

    @staticmethod
    def find_FootY(la, w, h):
        a = np.array(la) # l heel
        lh = np.multiply(a, [w, h]).astype(int)

        return lh[1]

    @staticmethod
    def find_FrontNeckY(a, b, w, h):
        a = np.array(a) # nose
        b = np.array(b) # should

        a2 = np.multiply(a, [w, h]).astype(int)
        b2 = np.multiply(b, [w, h]).astype(int)

        p = b2.copy() # should
        ydiff = abs(a2[1] - b2[1])
        y2 = (ydiff)/2
        p[1] = b2[1] - y2

        return p[1]

    @staticmethod
    def find_SideNeckY(a, b, c, w, h):
        a = np.array(a) # nose
        b = np.array(b) # l should
        c = np.array(c) # mouth r

        d_mouth_nose = abs(c[1] - a[1])

        n = np.multiply(a, [w, h]).astype(int)
        ls = np.multiply(b, [w, h]).astype(int)
        mr = np.multiply(c, [w, h]).astype(int)

        # l neck point
        lneck = ls.copy() # l neck
        ydiff = abs(n[1] - ls[1])
        y2 = (ydiff)//2
        lneck[1] = ls[1] - y2

        # r neck point
        d_mouth_nose = abs(mr[1] - n[1])

        rneck = lneck.copy()
        rneck[1] += d_mouth_nose

        return lneck[1], rneck[1]

    @staticmethod
    def find_ChestY(a, b, w, h):
        a = np.array(a) # elbow
        b = np.array(b) # should

        a2 = np.multiply(a, [w, h]).astype(int)
        b2 = np.multiply(b, [w, h]).astype(int)
            
        p = a2.copy() # elbow
        ydiff = abs(a2[1] - b2[1])
        y2 = (ydiff)/2
        p[1] = b2[1] + y2

        return p[1]

    @staticmethod
    def calculate_Chest(lel, lsh, rel, rsh, w, h):
        a = np.array(lsh) # l should
        b = np.array(rsh) # r should

        ls = np.multiply(a, [w, h]).astype(int) # l should
        rs = np.multiply(b, [w, h]).astype(int) # r should

        LY = MUtils.find_ChestY(lel, lsh, w, h)
        RY = MUtils.find_ChestY(rel, rsh, w, h)

        ls[1] = LY
        rs[1] = RY

        dist = np.linalg.norm(ls - rs)

        return dist

    @staticmethod
    def calculate_Waist(la, ra, lb, rb,  w, h):
        a = np.array(la) # l should
        b = np.array(ra) # r should
        c = np.array(lb) # l hip
        d = np.array(rb) # r hip

        ls = np.multiply(a, [w, h]).astype(int) # l should
        rs = np.multiply(b, [w, h]).astype(int) # r should
        lh = np.multiply(c, [w, h]).astype(int) # l hip
        rh = np.multiply(d, [w, h]).astype(int) # r hip

        lw = np.copy(ls)
        lw[0] = (abs(ls[0] - lh[0])) /2 + lh[0]
        lw[1] = (abs(lh[1] - ls[1])) /2 + ls[1]

        rw = np.copy(rs)
        rw[0] = (abs(rh[0] - rs[0])) /2 + rs[0]
        rw[1] = (abs(rh[1] - rs[1])) /2 + rs[1]

        dist = np.linalg.norm(lw - rw)

        return dist

    @staticmethod
    def find_Front_WaistY(la, ra, lb, rb,  w, h):
        a = np.array(la) # l should
        b = np.array(ra) # r should
        c = np.array(lb) # l hip
        d = np.array(rb) # r hip

        ls = np.multiply(a, [w, h]).astype(int) # l should
        rs = np.multiply(b, [w, h]).astype(int) # r should
        lh = np.multiply(c, [w, h]).astype(int) # l hip
        rh = np.multiply(d, [w, h]).astype(int) # r hip

        lw = np.copy(ls)
        lw[0] = (abs(ls[0] - lh[0])) /2 + lh[0]
        lw[1] = (abs(lh[1] - ls[1])) /2 + ls[1]

        rw = np.copy(rs)
        rw[0] = (abs(rh[0] - rs[0])) /2 + rs[0]
        rw[1] = (abs(rh[1] - rs[1])) /2 + rs[1]

        # dist = np.linalg.norm(lw - rw)

        return lw[1], rw[1]
    
    @staticmethod
    def find_Side_WaistY(la, ra, lb, rb,  w, h):
        a = np.array(la) # l should
        b = np.array(ra) # r should
        c = np.array(lb) # l hip
        d = np.array(rb) # r hip

        ls = np.multiply(a, [w, h]).astype(int) # l should
        rs = np.multiply(b, [w, h]).astype(int) # r should
        lh = np.multiply(c, [w, h]).astype(int) # l hip
        rh = np.multiply(d, [w, h]).astype(int) # r hip

        lw = np.copy(ls)
        lw[1] = (abs(lh[1] - ls[1])) /2 + ls[1]

        rw = np.copy(rs)
        rw[1] = (abs(rh[1] - rs[1])) /2 + rs[1]

        p = lw.copy() # l waist

        return p[1]

    @staticmethod
    def find_HipY(a, w, h):
        ar = np.array(a) # First
        a2 = np.multiply(ar, [w, h]).astype(int)

        return a2[1]

    @staticmethod
    def calculate_limb_dist(a, b, c, w, h, d=None):
        a = np.array(a) # First
        b = np.array(b) # Mid
        c = np.array(c) # End

        dist = None

        if d is None:
            a2 = np.multiply(a, [w, h]).astype(int)
            b2 = np.multiply(b, [w, h]).astype(int)
            c2 = np.multiply(c, [w, h]).astype(int)
            
            dist1 = np.linalg.norm(a2 - b2)
            dist2 = np.linalg.norm(b2 - c2)
            dist = abs(dist1) + abs(dist2)
        else:
            d = np.array(d)
            a2 = np.multiply(a, [w, h]).astype(int)
            b2 = np.multiply(b, [w, h]).astype(int)
            c2 = np.multiply(c, [w, h]).astype(int)
            d2 = np.multiply(c, [w, h]).astype(int)

            dist1 = np.linalg.norm(a2 - b2)
            dist2 = np.linalg.norm(b2 - c2)
            dist3 = np.linalg.norm(c2 - d2)
            dist = abs(dist1) + abs(dist2)+ abs(dist3)

        return dist

    @staticmethod
    def calculate_Perimeter(ancho: float, profundidad: float):  # Revisar resultados
        a = ancho/2
        b = profundidad/2
        
        perimeter = pi * ( 3*(a + b) - sqrt( (3*a + b) * (a + 3*b) ))
        return perimeter

    # CONTOUR RELATED FUNCTIONS
    @staticmethod
    def findMainContour(contours):
        """find main contour (2nd biggest contour area)"""
        areas = []
        for cont in contours:
            areas.append(cv2.contourArea(cont))

        n = len(areas)
        areas.sort()
        return contours[areas.index(areas[n-1])]

    
    @staticmethod
    def get_Xpts(contour, Ypoint):
        """find Xpts in contour for a given Ypos, conventional method"""
        xs = []
        for point in contour:
            x, y = point[0][:]
            if y == Ypoint:
                xs.append( (x, y) )

        lst = list(set(xs))
        lst.sort()
        return lst
    
    @staticmethod
    def get2Points(lst):
        """get first and last (x,y) coordinates from list of intersecting points in contour"""
        first, last = lst[0], lst[-1]

        first, last = [first[0], first[1]] , [last[0], last[1]]
        return first, last

    @staticmethod
    def getWaistPoints(LwaistPts, RwaistPts, imgWidth):
        minLDist = imgWidth # max length
        minRDist = imgWidth # max length
        Lpoint, Rpoint = (0,0), (0,0)
        mid = imgWidth//2 # mid

        for pnt in LwaistPts:
            x, y = pnt[:] # point
            if x > mid: continue
            dist = abs(mid - x)
            if dist < minLDist:
                minLDist = dist
                Lpoint = (x, y)

        for pnt in RwaistPts:
            x, y = pnt[:] # point
            if x < mid: continue
            dist = abs(mid - x)
            if dist < minRDist:
                minRDist = dist
                Rpoint = (x, y)

        waistPt1, waistPt2 = [Rpoint[0], Rpoint[1]] , [Lpoint[0], Lpoint[1]]
        return waistPt1, waistPt2

    @staticmethod
    def get_TopY(contour, Xpoint):
        """calc Ypos for the top of the head"""
        ys = []
        for point in contour:
            x, y = point[0][:]
            if x == Xpoint:
                ys.append( (x, y) )

        lst = list(set(ys))
        lst.sort()
        y = lst[0][1]

        return [Xpoint, y]

    @staticmethod
    def calculate_Height(contour, footY, imgWidth):
        """calculate Height with footY and headY(getTopY)"""
        midX = imgWidth//2
        bottomPoint = [midX, footY]
        topPoint = MUtils.get_TopY(contour, midX)

        a = np.array(bottomPoint) # p1
        b = np.array(topPoint) # p2

        dist = np.linalg.norm(a - b)
        # cv2.line(canvas, a, b, (255,0,0), 2, cv2.LINE_AA)

        return dist
        
    @staticmethod
    def calculate_Distance(pt1, pt2):
        '''makes numpy [x y] array from an [x,y] list to calc dist'''
        a = np.array(pt1) # p1
        b = np.array(pt2) # p2

        dist = np.linalg.norm(a - b)

        return dist

    @staticmethod
    def getNeckpoints(LNeckPts, RNeckPts):
        l1 = LNeckPts[0]
        r2 = RNeckPts[-1]

        l1, r2 = [l1[0], l1[1]] , [r2[0], r2[1]]
        return l1, r2

class PositionsFront():
    def __init__(self, neckY, LwaistY, RwaistY, hipY, footY, chest_dist_front, L_arm, R_arm, L_leg, R_leg):
        self.neckY = neckY
        self.LwaistY = LwaistY
        self.RwaistY = RwaistY
        self.hipY = hipY
        self.footY = footY
        self.chest_dist_front = chest_dist_front
        self.L_arm = L_arm
        self.R_arm = R_arm
        self.L_leg = L_leg
        self.R_leg = R_leg
        

class PositionsSide():
    def __init__(self, LneckY, RneckY, chestY, waistY, hipY):
        self.LneckY = LneckY
        self.RneckY = RneckY
        self.chestY = chestY
        self.waistY = waistY
        self.hipY = hipY


class MeasurementsFront():
    def __init__(self, distNeck, dist_Chest, distWaist, distHip, person_height, dist_Arm, dist_Leg):
        self.FneckDist = distNeck
        self.FchestDist = dist_Chest
        self.FwaistDist = distWaist
        self.FhipDist = distHip
        self.Height = person_height
        # left arm and leg
        self.FLarmDist = dist_Arm
        self.FLlegDist = dist_Leg


class MeasurementsSide():
    def __init__(self, distNeck, dist_Chest, distWaist, distHip):
        self.SneckDist = distNeck
        self.SchestDist = dist_Chest
        self.SwaistDist = distWaist
        self.ShipDist = distHip


class IMGSProcessor():
    def __init__(self, frontImg, sideImg):
        self.frontIMG = frontImg
        self.sideIMG = sideImg
        self.positionsFront =  self.find_positions_front()
        self.positionsSide = self.find_positions_side()

    def process_measurements(self):
        return self.find_contours_front(), self.find_contours_side()

    def find_positions_front(self):
        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as poser:
            img_front = self.frontIMG.copy()
            # img_front = MUtils.image_resize(img_front, height=730)  # TESTING
            h, w, _  = img_front.shape
            
            img_front = cv2.cvtColor(img_front, cv2.COLOR_BGR2RGB)
            results = poser.process(img_front)
            img_front = cv2.cvtColor(img_front, cv2.COLOR_RGB2BGR)
            
            # GET LANDMARKS
            landmarks = results.pose_landmarks.landmark
            l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            l_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
            r_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y]

            # Find positions
            neckY = MUtils.find_FrontNeckY(nose, l_shoulder, w, h)
            LwaistY, RwaistY = MUtils.find_Front_WaistY(l_shoulder, r_shoulder, l_hip, r_hip,  w, h)
            hipY = MUtils.find_HipY(l_hip, w, h)
            footY = MUtils.find_FootY(l_heel, w, h)

            # Calculate lengths, chest/leg/arm sizes
            chest_dist_front = MUtils.calculate_Chest(l_elbow, l_shoulder, r_elbow, r_shoulder, w, h)
            
            l_arm = MUtils.calculate_limb_dist(l_shoulder, l_elbow, l_wrist, w, h)
            r_arm = MUtils.calculate_limb_dist(r_shoulder, r_elbow, r_wrist, w, h)
            l_leg = MUtils.calculate_limb_dist(l_hip, l_knee, l_ankle, w, h, d=l_heel)
            r_leg = MUtils.calculate_limb_dist(r_hip, r_knee, r_ankle, w, h, d=r_heel)

        return PositionsFront(neckY, LwaistY, RwaistY, hipY, footY, chest_dist_front, l_arm, r_arm, l_leg, r_leg)

    def find_positions_side(self):
        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as poser:
            img_side = self.sideIMG.copy()
            # img_side = MUtils.image_resize(img_side, height=730)  # TESTING
            h, w, _  = img_side.shape

            # Recolor image to RGB
            img_side = cv2.cvtColor(img_side, cv2.COLOR_BGR2RGB)
            results = poser.process(img_side)
            img_side = cv2.cvtColor(img_side, cv2.COLOR_RGB2BGR)
            
            # GET LANDMARKS
            landmarks = results.pose_landmarks.landmark
            l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y]
            mouth_r = [landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].x,landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].y]

            # Find positions
            LneckY, RneckY = MUtils.find_SideNeckY(nose, l_shoulder, mouth_r, w, h)
            chestY = MUtils.find_ChestY(l_elbow, l_shoulder, w, h)
            waistY = MUtils.find_Side_WaistY(l_shoulder, r_shoulder, l_hip, r_hip,  w, h)
            hipY = MUtils.find_HipY(l_hip, w, h)

        return PositionsSide(LneckY, RneckY, chestY, waistY, hipY)

    def find_contours_front(self):
        img_in = self.frontIMG
        img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
        # img_in = MUtils.image_resize(img_in, height=730)  # TESTING
        
        # Threshold. Set values equal to or above 220 to 0. Set values below 220 to 255.
        _, img_thres = cv2.threshold(img_in, 245, 255, cv2.THRESH_BINARY_INV)
        
        img_floodfill = img_thres.copy()
        
        # Flood filling. Size needs to be 2 pixels bigger than the image.
        h, w = img_thres.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)

        # Floodfill from point (0, 0)
        cv2.floodFill(img_floodfill, mask, (0,0), 255)

        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(img_floodfill)

        # Combine the two images to get the foreground.
        img_out = img_thres | im_floodfill_inv
        # im_out is of shape (w, h, _)

        ######## After countour processed image #########

        img2gray = img_out.copy()

        ## blurring with kernel 25 ##
        kernel = np.ones((5, 5), np.float32)/25
        img2gray = cv2.filter2D(img2gray, -1, kernel)

        #extract contours from thresholded image
        _, thresh = cv2.threshold(img2gray, 250, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        ### GET SIZES, return sizes originally written in canvas ###
        cnt = contours[-1]
        if len(contours) != 1:
            cnt = MUtils.findMainContour(contours)
        
        neckPts = MUtils.get_Xpts(cnt, self.positionsFront.neckY)
        hipPts = MUtils.get_Xpts(cnt, self.positionsFront.hipY)
        LwaistPts = MUtils.get_Xpts(cnt, self.positionsFront.LwaistY)
        RwaistPts = MUtils.get_Xpts(cnt, self.positionsFront.RwaistY)

        # INTERSECTION POINTS | F:front, Pt1:leftmost (x,y) point, Pt2:rightmost (x,y) point
        FneckPt1, FneckPt2 = MUtils.get2Points(neckPts)
        FwaistPt1, FwaistPt2 = MUtils.getWaistPoints(LwaistPts, RwaistPts, imgWidth=w)
        FhipPt1, FhipPt2 = MUtils.get2Points(hipPts)

        ### RESULTS of type "numpy.float64", get float val by "numpy.float64.item()" ###
        distNeck = MUtils.calculate_Distance(FneckPt1, FneckPt2)
        distChest = self.positionsFront.chest_dist_front
        distWaist = MUtils.calculate_Distance(FwaistPt1, FwaistPt2)
        distHip = MUtils.calculate_Distance(FhipPt1, FhipPt2)
        person_height = MUtils.calculate_Height(cnt, self.positionsFront.footY, imgWidth=w)
        distLarm = self.positionsFront.L_arm
        distLleg = self.positionsFront.L_leg

        return MeasurementsFront(distNeck, distChest, distWaist, distHip, person_height, distLarm, distLleg)

    def find_contours_side(self):
        img_in = self.sideIMG
        img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
        # img_in = MUtils.image_resize(img_in, height=730)  # TESTING
        
        # Threshold. Set values equal to or above 220 to 0. Set values below 220 to 255.
        _, img_thres = cv2.threshold(img_in, 245, 255, cv2.THRESH_BINARY_INV)
        
        img_floodfill = img_thres.copy()
        
        # Flood filling. Size needs to be 2 pixels bigger than the image.
        h, w = img_thres.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)

        # Floodfill from point (0, 0)
        cv2.floodFill(img_floodfill, mask, (0,0), 255)

        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(img_floodfill)

        # Combine the two images to get the foreground.
        img_out = img_thres | im_floodfill_inv
        # im_out is of shape (w, h, _)

        ######## After countour processed image #########

        img2gray = img_out.copy()

        ## blurring with kernel 25 ##
        kernel = np.ones((5, 5), np.float32)/25
        img2gray = cv2.filter2D(img2gray, -1, kernel)

        #extract contours from thresholded image
        _, thresh = cv2.threshold(img2gray, 250, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        ### GET SIZES, return sizes originally written in canvas ###
        cnt = contours[-1]
        if len(contours) != 1:
            cnt = MUtils.findMainContour(contours)

        LneckPts = MUtils.get_Xpts(cnt, self.positionsSide.LneckY)
        RneckPts = MUtils.get_Xpts(cnt, self.positionsSide.RneckY)
        chestPts = MUtils.get_Xpts(cnt, self.positionsSide.chestY)
        waistPts = MUtils.get_Xpts(cnt, self.positionsSide.waistY)
        hipPts = MUtils.get_Xpts(cnt, self.positionsSide.hipY)

        # INTERSECTION POINTS | S:side, Pt1:leftmost (x,y) point, Pt2:rightmost (x,y) point
        SneckPt1, SneckPt2 = MUtils.getNeckpoints(LneckPts, RneckPts)
        SchestPt1, SchestPt2 = MUtils.get2Points(chestPts)
        SwaistPt1, SwaistPt2 = MUtils.get2Points(waistPts)
        ShipPt1, ShipPt2 = MUtils.get2Points(hipPts)

        ### RESULTS of type "numpy.float64", get float val by "numpy.float64.item()" ###
        distNeck = MUtils.calculate_Distance(SneckPt1, SneckPt2)
        distChest = MUtils.calculate_Distance(SchestPt1, SchestPt2)
        distWaist = MUtils.calculate_Distance(SwaistPt1, SwaistPt2)
        distHip = MUtils.calculate_Distance(ShipPt1, ShipPt2)

        return MeasurementsSide(distNeck, distChest, distWaist, distHip)


class AllMeasurements():
    def __init__(self, MFront: MeasurementsFront, MSide: MeasurementsSide):    
        self.MFront = MFront
        self.MSide = MSide
        self.neck_perimeter = self.calc_neck_peri()
        self.chest_perimeter = self.calc_chest_peri()
        self.waist_perimeter = self.calc_waist_peri()
        self.hip_perimeter = self.calc_hip_peri()

    def calc_neck_peri(self):
        Fneck = self.MFront.FneckDist.item()
        Sneck = self.MSide.SneckDist.item()

        perimeter = MUtils.calculate_Perimeter(Fneck, Sneck)
        return perimeter

    def calc_chest_peri(self):
        Fchest = self.MFront.FchestDist.item()
        Schest = self.MSide.SchestDist.item()

        perimeter = MUtils.calculate_Perimeter(Fchest, Schest)
        return perimeter

    def calc_waist_peri(self):
        Fwaist = self.MFront.FwaistDist.item()
        Swaist = self.MSide.SwaistDist.item()

        perimeter = MUtils.calculate_Perimeter(Fwaist, Swaist)
        return perimeter

    def calc_hip_peri(self):
        Fhip = self.MFront.FhipDist.item()
        Ship = self.MSide.ShipDist.item()

        perimeter = MUtils.calculate_Perimeter(Fhip, Ship)
        return perimeter