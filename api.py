import cv2

from gcloud import storage
from oauth2client.service_account import ServiceAccountCredentials
import os
from flask import Flask, request, render_template, jsonify, make_response

import numpy as np


def credentials_info():
    credentials_dict = {
        # Credentials Information
    }

    credentials = ServiceAccountCredentials.from_json_keyfile_dict(
        credentials_dict
    )
    return credentials


# client = storage.Client(credentials=credentials, project="glossy-fastness-305315")
# # bucket = client.get_bucket('mybucket')
# # blob = bucket.blob('myfile')
# print("DONE")


app = Flask(__name__)


def drawline(img, pt1, pt2, color, thickness=1, style='dotted', gap=5):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    if style == 'dotted':
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    else:
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)
            i += 1


def drawpoly(img, pts, color, thickness=1, style='dotted', ):
    s = pts[0]
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        drawline(img, s, e, color, thickness, style)


def drawrect(img, pt1, pt2, color, thickness=1, style='dotted'):
    pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
    drawpoly(img, pts, color, thickness, style)

# Hex Code to RGB
def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


# Draw Boxes around the object set for tracking
def drawBox(img, bbox, description, price, color, shape):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

    if shape == 'rectangle':
        cv2.rectangle(img, (x, y), ((x + w), (y + h)), color, 1, 1)
    elif shape == 'dotted':
        drawrect(img, (x, y), (x + w, y + h), color, 1, 'dotted')

    cv2.putText(img, str(description), (x, y + h + 22), cv2.FONT_HERSHEY_DUPLEX, 0.8, color,
                2)
    cv2.putText(img, str(price) + "$", (x, y + h + 45), cv2.FONT_HERSHEY_DUPLEX, 0.8, color,
                2)


# Function which takes tracks object
def object_track(video, bbox, description, price, second_array, color, shape,bucket):
    cap = cv2.VideoCapture(video)
    tracker = cv2.legacy.MultiTracker_create()
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Conversion from base64 string to image array
    # im_bytes = base64.b64decode(image)
    # im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    # image = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)

    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_seconds = int(total_frames / fps)
    frame_list = [int((total_frames / total_seconds) * x) for x in second_array]
    tracking_objects = 0



    # Initialising variables to store the tracker time of the objects
    objects = len(bbox)

    objects_tracked = [0] * objects
    c, d = [0] * objects, [0] * objects
    objects_lost = [0] * objects
    objects_tracked_list = []
    total_frame_count = 0

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)


    # Writing video to the file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    result = cv2.VideoWriter('filename.mp4',
                             fourcc,
                             10, (int(width), int(height)))

    new_list = []
    tracker_time = []

    while cap.isOpened():
        success, img = cap.read()
        if success:

            total_frame_count += 1
            if tracking_objects < len(second_array) and total_frame_count == frame_list[tracking_objects]:
                objects_tracked_list.append([])
                tracker_i = cv2.legacy_TrackerCSRT.create()
                # print(bbox)
                tracker.add(tracker_i, img, bbox[tracking_objects])
                tracking_objects += 1
            # Updating the tracker
            success, bboxes = tracker.update(img)

            # Iterating over the bounding boxes to calculate tracking time and drawing boxes around the object
            for i in range(len(bboxes)):

                if sum(bboxes[i]) != 0:
                    objects_tracked[i] += 1

                    drawBox(img, bboxes[i], description[i], price[i], hex_to_rgb(color[i]), shape[i])
                    c[i] += 1
                    if c[i] == 1:
                        objects_tracked_list[i].append(objects_lost[i])
                        d[i] = 0
                else:
                    objects_lost[i] += 1
                    d[i] += 1
                    if d[i] == 1:
                        objects_tracked_list[i].append(objects_tracked[i])
                        c[i] = 0

            result.write(img)
            cv2.waitKey(1)
        else:
            break
    # print(objects_tracked_list)
    # Calculating the total tracker objects tracked and objects lost
    for i in range(len(objects_tracked_list)):
        objects_tracked_list[i].append(total_frame_count)
        new_list.append([round(float(x + frame_list[i] / fps), 3) for x in objects_tracked_list[i]])

    # Adding all the elements with the previous values
    for i in range(len(new_list)):
        for j in range(1, len(new_list[i])):
            new_list[i][j] += new_list[i][j - 1]

    # Rounding the time into seconds along with previous time
    for i in range(len(new_list)):
        tracker_time.append([])
        for j in range(1, len(new_list[i]), 2):
            tracker_time[i].append(str(round(new_list[i][j - 1], 3)) + ":" + str(round(new_list[i][j], 3)))

    # print(tracker_time)
    cap.release()
    return tracker_time


@app.route('/object', methods=["POST"])
def object_tracking():
    if request.method == 'GET':
        return make_response(jsonify({'error': 'Please Send POST request'}))
    elif request.method == 'POST':

        params = request.get_json()
        credentials = credentials_info()
        client = storage.Client(credentials=credentials, project="glossy-fastness-305315")
        bucket = client.get_bucket('')
        blob = bucket.get_blob('/nich-performer-profile-p/610498e69013c44fa2a24591/61049ab69a0b640a5043110e/'+str(params['video']))
        blob.download_to_filename('sample.mp4')
        video = 'sample.mp4'

        # tracker_time returns the tracking time of the different objects tracked
        tracker_time = object_track(video, params['bbox'], params['description'],
                                    params['price'],
                                    params['seconds'],
                                    params['color'],
                                    params['shape'],bucket)

    return jsonify({"Tracker Time": tracker_time})


if __name__ == '__main__':
    app.run(debug=True)
