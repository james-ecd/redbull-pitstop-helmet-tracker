import cv2
import argparse
import numpy as np
import imutils
import json
import time
import uuid


class Helmet:

    def __init__(self, uid, center, x, y, radius, label):
        self.uuid = uid
        self.center = center
        self.x = x
        self.y = y
        self.radius = radius
        self.label = label
        self.centerX = center[0]
        self.centerY = center[1]
        self.saved = None
        self.accuracy = 15
        self.strayDistance = 50

    def __eq__(self, other):
        return self.centerX == other.centerX and self.centerY == other.centerY

    def __hash__(self):
        return hash(self.uuid)

    @property
    def hasMoved(self):
        if self.saved:
            return not (self.centerX - self.accuracy < self.saved.center[0] < self.centerX + self.accuracy and
                        self.centerY - self.accuracy < self.saved.center[1] < self.centerY + self.accuracy)
        else:
            return False

    def distanceFromLastRecorded(self, center):
        return [self.centerX - center[0], self.centerY - center[1]]

    @property
    def distanceFromLastSaved(self):
        return [abs(self.saved.center[0] - self.center[0]), abs(self.saved.center[1] - self.center[1])]

    def verifyNewDistance(self, center):
        return self.centerX - self.strayDistance < center[0] < self.centerX + self.strayDistance and \
               self.centerY - self.strayDistance < center[1] < self.centerY + self.strayDistance

    def isHelmet(self, center):
        return self.centerX - self.accuracy < center[0] < self.centerX + self.accuracy\
               and self.centerY - self.accuracy < center[1] < self.centerY + self.accuracy

    def savePosition(self):
        self.saved = Helmet(self.uuid, self.center, self.x, self.y, self.radius, self.label)

    def resetSaved(self):
        self.saved.update(self.center, self.x, self.y, self.radius)

    def update(self, center, x, y, radius):
        self.center = center
        self.x = x
        self.y = y
        self.radius = radius
        self.centerX = center[0]
        self.centerY = center[1]


class Tracker:

    MATCHING_METHODS = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
                        cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

    HELMET_LABELS = ['Barry', 'Lella', 'David', 'Annie', 'Paul', 'Tim', 'Gary', 'Sophie', 'Max', 'Daniel', 'Juan',
                     'James', 'Andy', 'Lewis', 'Jeroen', 'Formaggio', 'Jess']

    def getArguments(self):
        ap = argparse.ArgumentParser()
        ap.add_argument('-l', '--live', required=False,
                        help='Use live video source', action='store_true')
        ap.add_argument('-v', '--video', required=False,
                        help='Use pre-recorded video source')
        ap.add_argument('-r', '--roi', required=False,
                        help='Uses the region of interest specified in the setting.json file (generated with config script)',
                        action='store_true')
        ap.add_argument('-t', '--template', required=False,
                        help='Name of the template to use for tracking')
        ap.add_argument('-m', '--method', required=False,
                        help='Method to use in template matching')
        ap.add_argument('-d', '--debug', required=False,
                        help='Toggles debugging features (shows masks etc)', action='store_true')
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
        self.firstRun = True
        self.template = self.args.get('template', False)
        self.method = self.args.get('method', False)
        self.helmets = {}
        self.saved = False
        self.begin = False

    def findHelmets(self, frame, mask, label):
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        helmets = list(self.helmets.values())
        extraHelmets = []

        if len(cnts) > 0:
            count = 0
            for c in cnts:
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"])+1, int(M["m01"] / M["m00"])+1)

                if 10 < radius:
                    if self.begin:
                        if self.firstRun:
                            uid = str(uuid.uuid4())
                            self.helmets[uid] = Helmet(uid, center, x, y, radius, self.HELMET_LABELS[count])
                        else:
                            for h in helmets:
                                if h.isHelmet(center):
                                    if h.hasMoved:
                                        self.drawMovedHelmet(frame, h, center, x, y, radius)
                                    else:
                                        self.drawCircle(frame, center, x, y, radius, h.label)
                                    # Helmet found, remove from running list of unfound helmets and draw on frame
                                    helmets = [i for i in helmets if i.uuid != h.uuid]
                                    break
                            else:
                                # No helmet found with exact position
                                extraHelmets.append([center, x, y, radius])
                    else:
                        self.drawCircle(frame, center, x, y, radius, "")
                    count += 1
            if extraHelmets:
                # Some helmets have moved. Lets match them and draw the necessary extras
                for h in extraHelmets:
                    try:
                        distances = [(leftOver, leftOver.distanceFromLastRecorded(h[0])) for leftOver in helmets]
                        helmetMatch = min(distances, key=lambda i: i[1][0] + i[1][1])
                        if helmetMatch[0].verifyNewDistance(h[0]):
                            helmets = [i for i in helmets if i.uuid != helmetMatch[0].uuid]
                            print("Moved helmet detected and match with a x, y change of {}, {}".format(helmetMatch[1][0], helmetMatch[1][1]))
                            self.drawMovedHelmet(frame, self.helmets[helmetMatch[0].uuid], h[0], h[1], h[2], h[3])
                            self.helmets[helmetMatch[0].uuid].update(h[0], h[1], h[2], h[3])
                    except Exception as e:
                        print("Contours => helmet mismatch")
                        #raise e
            if self.begin and self.firstRun:
                self.firstRun = False

    def drawCircle(self, frame, center, x, y, radius, label, centerOnly=False, colour=(0,255,0)):
        if not centerOnly:
            cv2.circle(frame, (int(x), int(y)), int(radius), colour, 2)
        cv2.circle(frame, center, 5, (0, 0, 255), -1)
        cv2.putText(frame, label, (center[0] + 10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 3)

    # TODO: bounding boxes

    def drawMovedHelmet(self, frame, helmet, center, x, y, radius):
        if self.saved:
            self.drawCircle(frame, center, x, y, radius, '{} INCORRECT POSITION'.format(helmet.label), colour=(0,0,255))
            self.drawCircle(frame, helmet.saved.center, helmet.saved.x, helmet.saved.y, helmet.saved.radius,
                            '{} CORRECT POSITION'.format(helmet.label), colour=(0,255,0))
            self.drawGuidline(frame, helmet, center)
        else:
            self.drawCircle(frame, center, x, y, radius, helmet.label, colour=(0,255,0))

    def drawGuidline(self, frame, helmet, center):
        cv2.line(frame, helmet.saved.center, center, (0, 0, 0), 3)
        dist = helmet.distanceFromLastSaved
        lineXNegative = (helmet.saved.center[0] - center[0]) < 0
        lineYNegative = (helmet.saved.center[1] - center[1]) < 0
        lineCenterX = int(((helmet.saved.center[0] + center[0]) / 2) + 10)
        lineCenterY = int(((helmet.saved.center[1] + center[1]) / 2) - 10)
        if lineXNegative:
            promtX = "{}mm right".format(dist[0])
        else:
            promtX = "{}mm left".format(dist[0])
        if lineYNegative:
            promtY = "{}mm down".format(dist[1])
        else:
            promtY = "{}mm up".format(dist[1])
        cv2.putText(frame, "({},{})".format(promtX, promtY), (lineCenterX, lineCenterY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def saveHelmetPositions(self):
        for h in self.helmets.values():
            h.savePosition()
        self.saved = True
        print("Locations of {} helmets have been saved".format(len(self.helmets.values())))

    def findLogo(self, frame_rgb, method):
        template = cv2.imread(self.template, 0)
        frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
        w, h = template.shape[::-1]

        res = cv2.matchTemplate(frame_rgb, template, self.MATCHING_METHODS[int(method)])
        threshold = 0.8
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(frame_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    def resetHelmets(self):
        for h in self.helmets.values():
            h.resetSaved()
        print('{} helmets saved positions reset'.format(len(self.helmets.values())))

    def processFrame(self, frame):
        frame = imutils.resize(frame, width=1400)
        if self.args.get('roi'):
            r = self.roi
            frame = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        originialFrame = frame.copy()
        blur = cv2.GaussianBlur(frame, (15, 15), 0)
        r = self.roi

        if self.hsv:
            helmetFilter = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV).copy()
        elif self.rgb:
            helmetFilter = blur.copy()
        else:
            raise NotImplementedError('Only HSV or RGB filters are supported. Please use one of these')

        helmetMask = cv2.inRange(helmetFilter, self.helmetLower, self.helmetHigher)

        helmetMask = cv2.erode(helmetMask, None, iterations=4)
        helmetMask = cv2.dilate(helmetMask, None, iterations=2)

        if self.debug: cv2.imshow("mask", helmetMask)

        if self.template and self.method:
            self.findLogo(frame, self.method)
        #elif self.template ^ self.method:
            #raise Exception("Only one of template/method given. Both are required")
        else:
            self.findHelmets(frame, helmetMask, 'CREW')

        if self.debug:
            cv2.imshow("Helmet", helmetMask)

        return frame, originialFrame

    def video(self):
        # Ball tracking using a pre-recorded video source
        stream = cv2.VideoCapture(self.args['video'])

        while True:
            grabbed, frame = stream.read()
            if not grabbed:
                break
            frame, originialFrame = self.processFrame(frame)
            cv2.imshow("Frame", frame)
            if self.debug: cv2.imshow("Original", originialFrame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                self.saveHelmetPositions()
            if key == ord("b"):
                self.begin = True
            if key == ord("r"):
                self.resetHelmets()
            if key == ord("q"):
                break
        stream.release()

    def live(self):
        # Ball tracking using a live video source
        camera = cv2.VideoCapture(0)

        while True:
            ret, frame = camera.read()
            frame, originialFrame = self.processFrame(frame)
            cv2.imshow("Frame", frame)
            if self.debug: cv2.imshow("Original", originialFrame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                self.saveHelmetPositions()
            if key == ord("b"):
                self.begin = True
            if key == ord("r"):
                self.resetHelmets()
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
    tracker = Tracker()
    tracker.run()
