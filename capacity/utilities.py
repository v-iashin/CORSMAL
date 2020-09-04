import cv2
import shutil
import os

def extract_frames (object_set, modality_set, frame):
    for object in object_set:
        for modality in modality_set:
            if modality=='rgb' or modality=='ir':
                extract_frames_rgb(object, modality, frame)
            elif modality == 'depth':
                print('***************DEPTH **********************')
                extract_frames_depth(object, modality, frame)

def extract_frames_rgb(object, modality, frame):
    videos_path = os.getcwd() + '/video_database/' + object + '/' + modality + '/'
    frames_path = os.getcwd() + '/dataset/images/' + object + '/' + frame + '/'
    if not os.path.exists(frames_path):
        os.mkdir(frames_path)
    for filename in os.listdir(videos_path):
        #AVOID PUTTING THE .MP4 EXTENSION IN THE PATH
        save_image_path = frames_path + 'id' + object + '_' + filename[0:-4] + '_' + modality + '.png'
        if os.path.exists(save_image_path):
            print(save_image_path + ' ---- EXISTS ')
            continue
        vidcap = cv2.VideoCapture(videos_path + filename)
        print(filename + ' --- LOADING ')
        #To extract last frame
        # vidcap.set(1, vidcap.get(7) - 5)
        # To extract 20th to last frame
        if frame == '20':
            vidcap.set(1, vidcap.get(7)-20)
        success, image = vidcap.read()
        if success:
            cv2.imwrite(save_image_path, image)
            print(save_image_path + ' ---- SUCCESS ')
        vidcap.release()



def extract_frames_depth(object, modality, frame):
    #TODO to modify if possible to make it easier
    videos_path = os.getcwd() + '/video_database/' + object + '/' + modality + '/'
    frames_path = os.getcwd() + '/dataset/images/' + object + '/' + frame + '/'
    for root, first_level_dirs, first_level_files in os.walk(videos_path):
        for first_level_dir in first_level_dirs:
            first_level_joined_path = os.path.join(root, first_level_dir)
            for second_root, second_level_dirs, second_level_files in os.walk(first_level_joined_path):
                for second_level_dir in second_level_dirs:
                    second_level_joined_path = os.path.join(second_root, second_level_dir)
                    filename_list = os.listdir(second_level_joined_path)
                    if frame == '1':
                        filename = filename_list[0]
                    elif frame == '20':
                        filename = filename_list[-20]
                    # filename = filename_list[-5]
                    # filename = max(filename_list)
                    # for filename in os.listdir(second_level_joined_path):
                    #     if filename == '0000.png':
                    original_path = root + first_level_dir + '/' + second_level_dir + '/' + filename
                    to_move_path = frames_path + 'id' + object + '_' + first_level_dir + '_' + second_level_dir + '_depth.png'
                    if os.path.exists(to_move_path):
                        print(to_move_path + ' --- EXISTS')
                        continue
                    shutil.copyfile(original_path, to_move_path)
                    print(to_move_path + ' --- COPIED')



