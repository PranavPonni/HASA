import util
import visualizer as vis
from vis_tac import DualXelaVisualizer
import argparse
import os
import numpy as np
import pdb
import cv2



def main():
    parser = argparse.ArgumentParser(description='Write the episode number you want to visualize')
    parser.add_argument('--num', type=str, help='Integer number episode you want to visualize')

    args = parser.parse_args()

    # dir=os.path.join("/home/handling04/Documents/HASA/data_server/selftouch","episode"+args.num)
    dir="/home/handling04/Documents/HASA/data_server/stiff_pd/motion_all/0.7_-20_episode0"
    
    data=util.get_episode(dir)
    pdb.set_trace()
    # graph_to_video = vis.GraphToVideo()
    # graph_to_video.add_graph({"tactile_index_tip": data["tactile_index_tip"].reshape(-1,90),\
    #                           "tactile_thumb_tip": data["tactile_thumb_tip"].reshape(-1,90)})
    # graph_to_video.save_graph(".")
    tac_vis=DualXelaVisualizer()
    tac_vis.export_video_from_array(np.stack([data["tactile_index_tip"],data["tactile_thumb_tip"]],axis=1)/1e5)



if __name__ == '__main__':
    main()




