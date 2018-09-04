import cv2
import argparse
import numpy as np
import imutils
import json
import time
import uuid


class GameManager:

    def __init__(self):
        self.firstRun = True



class Helmet:

    def __init__(self, uid, center, radius, label):
        self.uuid = uid
        self.centerX = center[0]
        self.centerY = center[1]
        self.radius = radius

    def __eq__(self, other):
        return self.centerX == other.centerX and self.centerY == other.centerY

    def __hash__(self):
        return hash(self.uuid)

    def isBall(self, newCenter):
        mg = 20
        if missing:
            return self.centerX - mg <= newCenter[0] <= self.centerX + mg and self.centerY - mg <= newCenter[1] <= self.centerY + mg
        return self.centerX-2 <= newCenter[0] <= self.centerX+2 and self.centerY-2 <= newCenter[1] <= self.centerY+2

    def update(self, center):
        self.centerX = center[0]
        self.centerY = center[1]


class Game:

    def getArguments(self):
        ap = argparse.ArgumentParser()
        ap.add_argument('-l', '--live', required=False,
                        help='Use live video source', action='store_true')
        ap.add_argument('-v', '--video', required=False,
                        help='Use pre-recorded video source')
        ap.add_argument('-d', '--debug', required=False,
                        help='Toggles debugging features (shows masks etc)', action='store_true')
        ap.add_argument('-r', '--roi', required=False,
                        help='Uses the region of interest specified in the setting.json file (generated with config script)',
                        action='store_true')
        args = vars(ap.parse_args())

        # TODO - Add checks for file extensions
        return args

    def loadSettings(self):
        with open('settings.json') as f:
            data = json.load(f)
        return data['rgb'], data['hsv'], data['helmet_mask'], data['roi']

    def __init__(self):
        self.rgb, self.hsv, self.helmet, self.roi = self.loadSettings()
        self.helmetLower = tuple(self.helmet[:3])
        self.helmetHigher = tuple(self.helmet[3:])
        self.args = self.getArguments()
        self.debug = self.args['debug']
        self.gameManager = GameManager()
        self.firstRun = True

    def findHelmets(self, frame, mask, label):
        labelColours = {'white': (225, 225, 225), 'black': (0, 0, 0), 'yellow': (225, 225, 0), 'red': (0, 0, 225)}
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        center = None

        if len(cnts) > 0:
            for c in cnts:
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"])+1, int(M["m01"] / M["m00"])+1)

                if 20 < radius:
                    self.drawCircle(frame, center, x, y, radius, label)

    def drawCircle(self, frame, center, x, y, radius, label):
        if self.args.get('roi'):
            r = self.roi
            cv2.circle(frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])], (int(x), int(y)), int(radius),
                       (0, 255, 255), 2)
            cv2.circle(frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])], center, 5, (0, 0, 255), -1)
            cv2.putText(frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])], label, (center[0] + 10, center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        else:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            cv2.putText(frame, label, (center[0] + 10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)


    def processFrame(self, frame):
        frame = imutils.resize(frame, width=800)
        originialFrame = frame.copy()
        #kernel = np.ones((15, 15), np.float32) / 225
        #smoothed = cv2.filter2D(frame, -1, kernel)
        blur = cv2.GaussianBlur(frame, (15, 15), 0)
        r = self.roi

        if self.hsv:
            if self.args.get('roi'):
                helmetFilter = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV).copy()[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
            else:
                helmetFilter = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV).copy()
        elif self.rgb:
            if self.args.get('roi'):
                helmetFilter = blur.copy()[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
            else:
                helmetFilter = blur.copy()
        else:
            raise NotImplementedError('Only HSV or RGB filters are supported. Please use one of these')

        helmetMask = cv2.inRange(helmetFilter, self.helmetLower, self.helmetHigher)

        #helmetMask = cv2.erode(helmetMask, None, iterations=2)
        helmetMask = cv2.dilate(helmetMask, None, iterations=2)

        cv2.imshow("mask", helmetMask)

        self.findHelmets(frame, helmetMask, 'CREW')

        if self.debug:
            cv2.imshow("Helmet", helmetMask)

        return frame, originialFrame

    def video(self):
        # Ball tracking using a pre-recorded video source
        stream = cv2.VideoCapture(self.args['video'])
        time.sleep(2.0)

        while True:
            grabbed, frame = stream.read()
            if not grabbed:
                break
            frame, originialFrame = self.processFrame(frame)
            cv2.imshow("Frame", frame)
            cv2.imshow("Original", originialFrame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            time.sleep(0.05)
        stream.release()

    def live(self):
        # Ball tracking using a live video source
        camera = cv2.VideoCapture(0)

        while True:
            ret, frame = camera.read()
            frame, originialFrame = self.processFrame(frame)
            cv2.imshow("Frame", frame)
            cv2.imshow("Original", originialFrame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            time.sleep(0.05)

    def run(self):
        if self.args.get('video', False):
            self.video()
        elif self.args.get('live', False):
            self.live()
        else:
            raise ValueError('Either Image, Video or Webcam not specified. At-least one needed')
        cv2.destroyAllWindows()


if __name__ == '__main__':
    game = Game()
    game.run()
