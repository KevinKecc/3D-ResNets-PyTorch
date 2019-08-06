from __future__ import print_function, division
import os
import subprocess


if __name__=="__main__":
    root_path = '/data/Charades/Charades'
    dir_path = '/data/Charades/Charades_v1'
    dst_dir_path = '/data/Charades/Charades_jpg'

    vid_name_list = os.listdir(dir_path)

    for vid_name in vid_name_list:
        src_path = os.path.join(dir_path, vid_name)
        dst_path = os.path.join(dst_dir_path, vid_name.split('.')[0])
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        cmd = 'ffmpeg -loglevel 0 -i \"{}\"  \"{}/image_%05d.jpg\"'.format(src_path, dst_path)
        print(cmd)
        subprocess.call(cmd, shell=True)
        print('\n')








