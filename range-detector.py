import cv2
import json
import os
import argparse
import sys


class RangeDetector:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        self.args = self.get_arguments()
        self.filter = self.args['filter'].upper()
        self.vid = VideoCapture(self.args, self.camera)

    def run(self):
        while True:
            ret, images = self.vid.getFrame()
            if not ret:
                break
            cv2.imshow("Original", images[0])
            cv2.imshow("thresh", images[1])
            cv2.imshow("Preview", images[2])

            if cv2.waitKey(1) & 0xFF is ord('s'):
                self.saveSliderValues()
            if cv2.waitKey(1) & 0xFF is ord('l'):
                self.vid.loadSliderValues()
            if cv2.waitKey(1) & 0xFF is ord('q'):
                self.vid.release()
                cv2.destroyAllWindows()
                sys.exit()

    def get_arguments(self):
        ap = argparse.ArgumentParser()
        ap.add_argument('-f', '--filter', required=True,
                        help='Range filter. RGB or HSV')
        args = vars(ap.parse_args())

        if not args['filter'].upper() in ['RGB', 'HSV']:
            ap.error("Please speciy a correct filter.")

        return args

    def saveSliderValues(self):
        if os.path.isfile('settings.json'):
            with open('settings.json') as file:
                data = json.load(file)
            v1, v2, v3, v4, v5, v6 = self.vid.get_trackbar_values()
            data['helmet_mask'] = [v1, v2, v3, v4, v5, v6]
            if self.filter == 'RGB':
                data['rgb'] = True
                data['hsv'] = False
            else:
                data['rgb'] = False
                data['hsv'] = True
            with open('settings.json', 'w') as file:
                json.dump(data, file)
            print('\nMask values and filter type saved to setting file')
        else:
            print('\nEither settings file is missing or not readable...')


class VideoCapture:

    def __init__(self, args, source):
        self.args = args
        self.range_filter = self.args['filter'].upper()
        self.camera = source
        self.setup_trackbars(self.range_filter)

    def getFrame(self):
        ret, image = self.camera.read()
        if not ret:
            return False, None
        if self.range_filter == 'RGB':
            self.frame_to_thresh = image.copy()
        else:
            self.frame_to_thresh = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        v1_min, v2_min, v3_min, v1_max, v2_max, v3_max = self.get_trackbar_values()

        thresh = cv2.inRange(self.frame_to_thresh, (v1_min, v2_min, v3_min), (v1_max, v2_max, v3_max))

        preview = cv2.bitwise_and(image, image, mask=thresh)

        return True, [image, thresh, preview]

    def get_trackbar_values(self):
        values = []

        for i in ["MIN", "MAX"]:
            for j in self.range_filter:
                v = cv2.getTrackbarPos("%s_%s" % (j, i), "Trackbars")
                values.append(v)

        return values

    def loadSliderValues(self):
        if os.path.isfile('settings.json'):
            print('\nSettings file found and read')
            with open('settings.json') as file:
                data = json.load(file)
            sliderVals = data['helmet_mask']
            count = 0
            for i in ["MIN", "MAX"]:
                for j in self.range_filter:
                    cv2.setTrackbarPos("%s_%s" % (j, i), "Trackbars", sliderVals[count])
                    count += 1
            print("Loaded in values from settings file")
        else:
            print('\nEither settings file is missing or not readable...')

    def callback(self, value):
        pass

    def setup_trackbars(self, range_filter):
        cv2.namedWindow("Trackbars", 0)

        for i in ["MIN", "MAX"]:
            v = 0 if i == "MIN" else 255

            for j in range_filter:
                cv2.createTrackbar("%s_%s" % (j, i), "Trackbars", v, 255, self.callback)

    def release(self):
        self.camera.release()


if __name__ == '__main__':
    rangeDetector = RangeDetector()
    rangeDetector.run()
