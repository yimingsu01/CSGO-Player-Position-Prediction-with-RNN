import math
import cv2
import numpy as np

def graphpos(poses, orig_std, orig_mean, show, debug, filename):
    player_pos = np.array([[0, 0]])
    for i in range(0, 10, 2):
        temp = np.array([])
        temp = np.append(temp, poses[i])
        temp = np.append(temp, poses[i + 1])
        temp = np.expand_dims(temp, axis=0)

        player_pos = np.row_stack((player_pos, temp))

    player_pos = np.delete(player_pos, 0, axis=0)
    # print(player_pos)

    # data = (t_player_pos - np.mean(t_player_pos)) / np.std(t_player_pos)
    player_pos = player_pos * orig_std + orig_mean

    # new_value = ( (old_value - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min
    old_x_max = 2760
    old_x_min = -1872

    old_y_max = 3675
    old_y_min = -957

    new_x_max = 1024
    new_y_max = new_x_max

    new_x_min = 0
    new_y_min = new_y_max

    player_pos[:, 0] = ((player_pos[:, 0] - old_x_min) / (old_x_max - old_x_min)) * (1024)
    player_pos[:, 1] = ((player_pos[:, 1] - old_y_min) / (old_y_max - old_y_min)) * (1024)

    if debug == 1:
        print(player_pos)

    inf = cv2.imread("de_inferno.png")
    for pos in player_pos:
        cv2.circle(img=inf, center=(int(pos[0]), int(1024-pos[1])), radius=5, color=(0,255,255), thickness=-1)

    if show == 1:
        # Using cv2.imshow() method
        # Displaying the image
        cv2.imshow("res", inf)

        #waits for user to press any key
        #(this is necessary to avoid Python kernel form crashing)
        cv2.waitKey(0)

        #closing all open windows
        cv2.destroyAllWindows()
    else:
        cv2.imwrite(filename, inf)