import cv2

from gcloud import storage
from oauth2client.service_account import ServiceAccountCredentials
import os
from flask import Flask, request, render_template, jsonify, make_response
import io, base64
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


# Draw Boxes around the object set for tracking
def drawBox(img, bbox, description, price):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

    cv2.rectangle(img, (x, y), ((x + w), (y + h)), (255, 0, 255), 1, 1)

    cv2.putText(img, str(description), (x, y - 8), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 0, 255),
                1, lineType=cv2.LINE_AA)
    cv2.putText(img, str(price) + "$", (x, y + h + 22), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 255),
                1)


# Function which takes tracks object
def object_track(video, image, bboxes, description, price, objects):
    cap = cv2.VideoCapture(video)
    tracker = cv2.legacy.MultiTracker_create()

    # Conversion from base64 string to image array
    im_bytes = base64.b64decode(image)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    image = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)

    # Setting the tracker as per the number of object
    for i in range(objects):
        tracker_i = cv2.legacy.TrackerCSRT_create()
        bbox = bboxes[i]
        bb = tuple((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
        tracker.add(tracker_i, image, bb)

    # Initialising variables to store the tracker time of the objects
    objects_tracked = [0] * objects
    c, d = [0] * objects, [0] * objects
    objects_lost = [0] * objects
    objects_tracked_list = []
    total_frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)  # Provides the fps to calculate the tracker time of object

    # Creating empty lists to append the tracker time
    for i in range(2):
        objects_tracked_list.append([])

    # Writing video to the file
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    result = cv2.VideoWriter('filename.avi',
                             fourcc,
                             10, (image.shape[1], image.shape[0]))

    new_list = []
    tracker_time = []

    while cap.isOpened():
        success, img = cap.read()
        if success:

            total_frame_count += 1

            # Updating the tracker
            success, bboxes = tracker.update(img)

            # Iterating over the bounding boxes to calculate tracking time and drawing boxes around the object
            for i in range(len(bboxes)):

                if sum(bboxes[i]) != 0:
                    objects_tracked[i] += 1

                    drawBox(img, bboxes[i], description[i], price[i])
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
        new_list.append([round(float(x / fps), 3) for x in objects_tracked_list[i]])

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
        blob = bucket.get_blob(params['video'])
        blob.download_to_filename('sample.mp4')
        video = 'sample.mp4'

        # tracker_time returns the tracking time of the different objects tracked
        tracker_time = object_track(video, params['image'], params['bbox'], params['description'],
                                    params['price'],
                                    params['objects'])

    return jsonify({"Tracker Time": tracker_time})


if __name__ == '__main__':
    app.run(debug=True)
