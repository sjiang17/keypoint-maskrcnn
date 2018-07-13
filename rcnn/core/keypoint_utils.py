import numpy as np

def flip_keypoints(kps, width):

    fliped_kps = kps.copy()
    fliped_kps[:,0] = width - kps[:,0] - 1

    return fliped_kps

# def keypoints_to_map_wrt_box(keypoints, box, M):
#
#     return

def keypoints_to_vec_wrt_box(keypoints, box, M):
    # keypoints: (17, 3) matrix
    # (x, y, v), where v is the validity of the point
    def valid_coordinate(coord, M_SIZE):
        if coord[0] >=0 and coord[0] <= M_SIZE and coord[1] >=0 and coord[1] <= M_SIZE:
            return True
        return False

    kps = np.array(keypoints, dtype=np.int32)
    # print(kps)
    num_kp = kps.shape[0]
    assert num_kp == 17, 'invalid number of keypoints: {}'.format(num_kp)

    w = box[2] - box[0]
    h = box[3] - box[1]

    w = np.maximum(w, 1)
    h = np.maximum(h, 1)

    kps[:, 0] = ((kps[:, 0] - box[0]) * M / w).astype(np.int32)
    kps[:, 1] = ((kps[:, 1] - box[1]) * M / h).astype(np.int32)

    DEBUG = False
    if DEBUG:
        valid_cnt, invalid_cnt = 0, 0
        for i in range(17):
            if kps[i,2] == 0:
                if not valid_coordinate(kps[i, 0:2], M-1):
                    invalid_cnt += 1
            else:
                if not valid_coordinate(kps[i, 0:2], M-1):
                    valid_cnt += 1
        if valid_cnt + invalid_cnt < 17:
            print("True")
        else:
            print("False")
        # print("valid:{}, invalid:{}".format(valid_cnt, invalid_cnt))

    label_index_vec = np.full((num_kp,), -1)
    for i in range(num_kp):
        if int(kps[i, 2]) == 0:
            # the gt index label of invalid keypoint should be set to -1
            continue
        elif valid_coordinate(kps[i, 0:2], M-1):
            valid_index = kps[i, 0] + kps[i, 1] * M
            assert valid_index >= 0 and valid_index <= M*M-1
            label_index_vec[i] = valid_index

    return label_index_vec

