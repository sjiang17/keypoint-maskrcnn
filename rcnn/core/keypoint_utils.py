import numpy as np
import cv2

# def get_circle():
#     a, b = 1, 1
#     n = 7
#     r = 3
#
#     y, x = np.ogrid[-a:n - a, -b:n - b]
#     mask = x * x + y * y <= r * r
#
#     return mask

def get_keypoints():
    # Get the COCO keypoints and their left/right flip coorespondence map.

    keypoint_names = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    ]
    keypoint_flip_map = {
        'left_eye': 'right_eye',
        'left_ear': 'right_ear',
        'left_shoulder': 'right_shoulder',
        'left_elbow': 'right_elbow',
        'left_wrist': 'right_wrist',
        'left_hip': 'right_hip',
        'left_knee': 'right_knee',
        'left_ankle': 'right_ankle'
    }
    return keypoint_names, keypoint_flip_map

def get_skeletons():
    skeletons = [[15, 13],
                [13, 11],
                [16, 14],
                [14, 12],
                [11, 12],
                [5, 11],
                [6, 12],
                [5, 6],
                [5, 7],
                [6, 8],
                [7, 9],
                [8, 10],
                [1, 2],
                [0, 1],
                [0, 2],
                [1, 3],
                [2, 4],
                [3, 5],
                [4, 6]]
    return skeletons

def flip_keypoints(keypoints, width):
    # Left/right flip keypoint_coords.
    keypoint_names, keypoint_flip_map = get_keypoints()
    flipped_kps = keypoints.copy()
    for lkp, rkp in keypoint_flip_map.items():
        lid = keypoint_names.index(lkp)
        rid = keypoint_names.index(rkp)
        flipped_kps[:,lid,:] = keypoints[:,rid,:]
        flipped_kps[:,rid,:] = keypoints[:,lid,:]

    flipped_kps[:,:,0] = width - flipped_kps[:,:,0] - 1
    inds = np.where(flipped_kps[:,:,2] == 0)
    flipped_kps[inds[0],inds[1],0] = 0

    return flipped_kps

def valid_coordinate(coord, M):
    if coord[0] >=0 and coord[0] < M and coord[1] >=0 and coord[1] < M:
        return True
    return False

def pixel_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def distance_map(kp, M):
    distance_map = np.array([[np.linalg.norm([_x - kp[0], _y - kp[1]]) for _x in range(M)] for _y in range(M)])
    return distance_map


def keypoints_to_map_wrt_box(keypoints, box, M):
    # keypoints: (17, 3) matrix
    # (x, y, v), where v is the validity of the point
    kps = np.array(keypoints, dtype=np.int32)
    num_kp = kps.shape[0]
    assert num_kp == 17, 'invalid number of keypoints: {}'.format(num_kp)

    w = box[2] - box[0]
    h = box[3] - box[1]
    w = np.maximum(w, 1)
    h = np.maximum(h, 1)

    kps[:, 0] = ((kps[:, 0] - box[0]) * M / w).astype(np.int32)
    kps[:, 1] = ((kps[:, 1] - box[1]) * M / h).astype(np.int32)

    kp_target_map = np.full((17, M, M), -1, dtype=np.float32)
    for k in range(num_kp):
        if int(kps[k,2]) == 0:
            continue
        elif valid_coordinate(kps[k,0:2], M):
            kp_target_map[k, :, :] = 0
            kp_target_map[k, kps[k,0], kps[k,1]] = 1
            xl = kps[k, 0] - 1
            xr = kps[k, 0] + 1
            yu = kps[k, 1] - 1
            yd = kps[k, 1] + 1
            if xl >= 0:
                kp_target_map[k, xl, kps[k, 1]] = 1
            if xr < M:
                kp_target_map[k, xr, kps[k, 1]] = 1
            if yu >=0:
                kp_target_map[k, kps[k, 0], yu] = 1
            if yd < M:
                kp_target_map[k, kps[k, 0], yd] = 1

            # kp_target_map[k, d_map <= radius] = 1
            # kp_target_map[k, d_map > radius] = 0
    # print kp_target_map[0]
    # kp_target_map = np.full((17, M, M), 1, dtype=np.float32)
    return kp_target_map

def keypoints_to_gaussian_map_wrt_box(keypoints, box, M):
    # keypoints: (17, 3) matrix
    # (x, y, v), where v is the validity of the point
    kps = np.array(keypoints, dtype=np.int32)
    num_kp = kps.shape[0]
    assert num_kp == 17, 'invalid number of keypoints: {}'.format(num_kp)

    w = box[2] - box[0]
    h = box[3] - box[1]
    w = np.maximum(w, 1)
    h = np.maximum(h, 1)

    kps[:, 0] = ((kps[:, 0] - box[0]) * M / w).astype(np.int32)
    kps[:, 1] = ((kps[:, 1] - box[1]) * M / h).astype(np.int32)

    kp_target_map = np.full((17, M, M), 0, dtype=np.float32)
    for k in range(num_kp):
        if int(kps[k,2]) == 0:
            continue
        elif valid_coordinate(kps[k,0:2], M):
            kp_target_map[k, :, :] = 0
            kp_target_map[k, kps[k,0], kps[k,1]] = 1
            kp_target_map[k] = cv2.GaussianBlur(kp_target_map[k], (3,3), 0)
            am = np.amax(kp_target_map[k])
            kp_target_map[k] /= am / 1.0

    return kp_target_map

def keypoints_to_gaussian_map_and_weight_wrt_box(keypoints, box, M):
    # keypoints: (17, 3) matrix
    # (x, y, v), where v is the validity of the point
    kps = np.array(keypoints, dtype=np.int32)
    num_kp = kps.shape[0]
    assert num_kp == 17, 'invalid number of keypoints: {}'.format(num_kp)

    w = box[2] - box[0]
    h = box[3] - box[1]
    w = np.maximum(w, 1)
    h = np.maximum(h, 1)

    kps[:, 0] = ((kps[:, 0] - box[0]) * M / w).astype(np.int32)
    kps[:, 1] = ((kps[:, 1] - box[1]) * M / h).astype(np.int32)

    kp_target_map = np.full((17, M, M), -1, dtype=np.float32)
    kp_weight_map = np.zeros((17, M, M), dtype=np.float32)

    for k in range(num_kp):
        if int(kps[k,2]) == 0:
            continue
            # kp_weight_map[k,:,:] = 0
        elif valid_coordinate(kps[k,0:2], M):
            kp_target_map[k, :, :] = 0
            kp_target_map[k, kps[k,0], kps[k,1]] = 1
            kp_target_map[k] = cv2.GaussianBlur(kp_target_map[k], (3,3), 0)
            am = np.amax(kp_target_map[k])
            kp_target_map[k] /= am / 1.0
            kp_weight_map[k, :, :] = 1
    # print(kp_target_map[0])
    return kp_target_map, kp_weight_map


def keypoints_to_vec_wrt_box(keypoints, box, M):
    # keypoints: (17, 3) matrix
    # (x, y, v), where v is the validity of the point
    kps = np.array(keypoints, dtype=np.int32)
    num_kp = kps.shape[0]
    assert num_kp == 17, 'invalid number of keypoints: {}'.format(num_kp)

    w = box[2] - box[0]
    h = box[3] - box[1]
    w = np.maximum(w, 1)
    h = np.maximum(h, 1)

    kps[:, 0] = ((kps[:, 0] - box[0]) * M / w).astype(np.int32)
    kps[:, 1] = ((kps[:, 1] - box[1]) * M / h).astype(np.int32)

    label_index_vec = np.full((num_kp,), -1, dtype=np.int32)
    for i in range(num_kp):
        if int(kps[i, 2]) == 0:
            # the gt index label of invalid keypoint should be set to -1
            continue
        elif valid_coordinate(kps[i, 0:2], M):
            valid_index = kps[i, 0] + kps[i, 1] * M
            assert valid_index >= 0 and valid_index <= M*M-1

            label_index_vec[i] = valid_index

    return label_index_vec

def decode_keypoint(kp_label, bbox):
    x = int(kp_label) % 56
    y = int(kp_label) / 56

    x = bbox[0] + float(x) / 56.0 * (bbox[2] - bbox[0])
    y = bbox[1] + float(y) / 56.0 * (bbox[3] - bbox[1])
    return x, y

def get_keypoint_wrt_box(x_coord, y_coord, bbox):

    x = bbox[0] + float(x_coord) / 28.0 * (bbox[2] - bbox[0])
    y = bbox[1] + float(y_coord) / 28.0 * (bbox[3] - bbox[1])
    return x, y