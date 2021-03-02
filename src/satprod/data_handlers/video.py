import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import torchvision
import cv2
from img_data import SatImgDataset, SatImg
#from src.satprod.data_handlers.img_data import SatImgDataset

class SatVid():

    def __init__(self, root: str):
        self.root = root
        
        data = SatImgDataset(self.root)

        self.img_paths = data.img_paths
        self.timestamps = data.timestamps
        self.getDateIdx = data.getDateIdx

        self.videopath = os.path.join(root, 'data/video')
    
    def create(self, name: str, 
               start: datetime=datetime(2018,3,20,0),
               stop: datetime=datetime(2018,3,24,0),
               resize: bool=False,
               scale: int=5):

        self.start_idx = self.getDateIdx(start)
        self.stop_idx = self.getDateIdx(stop)

        img_array = []
        for i in range(self.start_idx, self.stop_idx):
            img = cv2.imread(self.img_paths[i])

            # downscale image
            if resize: 
                img = SatImg(img, self.timestamps[self.start_idx+i])
                img = img.resize(scale_percent=scale)
                img = img.img
            
            img_array.append(img)

        height, width, _ = img.shape
        size = (width,height)

        out = cv2.VideoWriter(os.path.join(self.videopath, name), cv2.VideoWriter_fourcc(*'DIVX'), 5, size)

        for img_obj in img_array:
            out.write(img_obj)
        out.release()

    def play(self, name: str, fps: int = 2):
        delay_time = int(1000/fps)

        cap = cv2.VideoCapture(os.path.join(self.videopath, name))

        while(True):
            # Capture frame-by-frame
            _, frame = cap.read()

            # Our operations on the frame come here
            if frame is not None: 
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                cap.release()
                cv2.destroyAllWindows()
                break

            # Display the resulting frame
            cv2.imshow('frame',gray)
            
            cv2.waitKey(delay_time)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
                #break
        cv2.destroyAllWindows()
    
    def delete(self, name: str):
        os.remove(os.path.join(self.videopath, name))

class OpticalFlow():

    def __init__(self, root: str, name: str):
        self.root = root
        self.name = name
        self.videopath = os.path.join(root, 'data/video')
    
    def dense(self, save: str):
        cap = cv2.VideoCapture(os.path.join(self.videopath, self.name))

        _, frame1 = cap.read()
        prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255

        counter = 0
        while(1):
            print(counter)
            counter += 1
            _, frame2 = cap.read()
            if frame2 is not None: 
                next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

                flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                hsv[...,0] = ang*180/np.pi/2
                hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
                rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

                cv2.imshow('frame2',rgb)
                #k = cv2.waitKey(30) & 0xff
                #if k == 27:
                    #break
                #elif k == ord('s'):
                cv2.imwrite(f'{save}/opticalfb_{counter}.png',frame2)
                cv2.imwrite(f'{save}/opticalhsv_{counter}.png',rgb)
                prvs = next
            else:
                #cv2.imwrite('opticalfb.png',frame2)
                #cv2.imwrite('opticalhsv.png',rgb)
                cap.release()
                cv2.destroyAllWindows()
                break

        cap.release()
        cv2.destroyAllWindows()

    def lukasKanade(self, save: str):
        cap = cv2.VideoCapture(os.path.join(self.videopath, self.name))

        # params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 100,
                            qualityLevel = 0.3,
                            minDistance = 7,
                            blockSize = 7 )

        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (5,5),#(15,15),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Create some random colors
        color = np.random.randint(0,255,(100,3))

        # Take first frame and find corners in it
        _, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)
        counter = 0
        while(1):
            print(counter)
            counter +=1
            _, frame = cap.read()
            if frame is not None:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # calculate optical flow
                p1, st, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

                # Select good points
                good_new = p1[st==1]
                good_old = p0[st==1]

                # draw the tracks
                for i,(new,old) in enumerate(zip(good_new,good_old)):
                    a,b = new.ravel()
                    c,d = old.ravel()
                    mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                    frame = cv2.circle(frame,(a,b),1,color[i].tolist(),-1)
                img = cv2.add(frame,mask)

                #cv2.imshow('frame',img)
                cv2.imwrite(f'{save}/opticalfeatures_{counter}.png',img)
                cv2.imwrite(f'{save}/opticalmask_{counter}.png',mask)

                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break

                # Now update the previous frame and previous points
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1,1,2)
            else:
                break

        cv2.destroyAllWindows()
        cap.release()


if __name__=='__main__':
    cd = str(os.path.dirname(os.path.abspath(__file__)))
    root = f'{cd}/../../..'
    v = SatVid(root)
    
    start = datetime(2018,6,28,6)
    stop = datetime(2018,6,28,18)

    name = f'resized.avi'
    #v.delete(name)

    v.create(name, start, stop, True, 20)
    v.play(name, 0.75)

    of = OpticalFlow(root, name)
    #of.lukasKanade(f'.')
    of.dense(f'.')

    v.delete(name)

    '''for i in range(5):
        start += timedelta(hours=1)
        stop = start + timedelta(hours=2)
        name = f'take_{i}.avi'
        v.delete(name)
        v.create(name, start, stop)
        #v.play(name, 0.5)

        of = OpticalFlow(root, name)
        os.makedirs(f'ex6/{i}', exist_ok=True)
        of.feature(f'ex6/{i}')
        #of.dense(f'ex6/{i}')
        v.delete(name)'''