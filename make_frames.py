import datetime
import pymongo
from collections import OrderedDict, defaultdict
import pandas as pd
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor
import urllib2
import os
import cv2
import math

myclient = pymongo.MongoClient(
    "mongodb://PBpyUser:PBPYPolicy@10.80.30.222:27017,10.80.40.97:27017,10.80.30.121:27017/PBpy?replicaSet=rs6&readPreference=secondaryPreferred&connectTimeoutms=6000")
db = myclient.PBpy
date = datetime.datetime.today() + relativedelta(days=-1)
from_date = datetime.datetime.combine(date, datetime.datetime.min.time())
# cursor = db.inspection.find({"InspectionDate":{"$gte":from_date,'$lt':date}}).limit(20)
cursor = db.inspection.find({},no_cursor_timeout=True).limit(3000).sort("$natural", -1)
path = "/home/tarun/ankit/arvind/videos/"
counter = 0

selected_cars = ["WAGON R", "SWIFT", "DZIRE", "i 20", "i 10", "i", "WAGON"]

for record in cursor:
    print (record)
    if record["Model"]:
    	model = record["Model"].split(" ")[0]
    else:
	model = None

    if ".zip" not in record["FileUrl"] and model and model in selected_cars:
        response = urllib2.urlopen(record["FileUrl"])
        dir_path = os.path.join(path, record['InspectionId'])
	print (dir_path)
        os.makedirs(dir_path)
        vid_path = os.path.join(dir_path, record['FileUrl'].split('/')[-2])
	print (vid_path)
        with open(vid_path, 'wb') as w:
            w.write(response.read())
        cap = cv2.VideoCapture(vid_path)
        frameRate = cap.get(cv2.CAP_PROP_FPS)  # frame rate
        totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        videoDuration = round(totalFrames / frameRate)
        skipTime = videoDuration - 150

        if skipTime > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, skipTime * 1000)
            print (counter)

            var_name = []
	    x = []
            while (cap.isOpened()):
                frameId = cap.get(cv2.CAP_PROP_POS_MSEC)  # current frame number
                print (frameId)
                ret, frame = cap.read()
                if (ret != True):
                    break
                if (math.floor(frameId) % math.floor(frameRate) == 0):
                    counter += 1
                    filename = str(int(counter)) + ".jpg"
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    var = cv2.Laplacian(gray, cv2.CV_64F).var()
                    var_name.append({"f_name":filename,"var":var, "frame":frame})
                    x.append(var)
            x.sort()
            sum_avg = round(sum(x[:5]) / 5)

        for image in var_name:
            if image["var"] > sum_avg:
                cv2.imwrite(os.path.join(dir_path, image["f_name"]), image["frame"])

    	cap.release()
    counter = 0
    print ("Done!")
