from py_node_exec import NodeExec
from allegro_package import AllegroHand
import time
import keyboard  
import numpy as np

if __name__ == "__main__":
    node = NodeExec(freq=10)
    node.spin_thread_start()
    
    robot = AllegroHand(hand_topic_prefix="allegroHand_0", ctrl_freq=5)
    
    state = "idle"           
    recorded_jnt_pos = []     
    play_idx = 0              
    
    print("Press 's' to start teaching (recording).")
    print("Press 'r' to run (playback).")
    print("Press 'q' to quit.")
    
    while node.ok():
        if keyboard.is_pressed('q'):
            break
        
        if state == "idle":
            if keyboard.is_pressed('s'):
                state = "recording"
                recorded_jnt_pos = []
                print("Recording started.")
                time.sleep(0.5)  
        elif state == "recording":
            if keyboard.is_pressed('r'):
                state = "playing"
                play_idx = 0
                print("Playback started.")
                time.sleep(0.5)
        
        obs = robot.get_obs()
        
        if state == "recording":
            recorded_jnt_pos.append(obs["jnt_pos"])
            robot.set_joint_cmd(obs["jnt_pos"])
        elif state == "playing":
            if play_idx < len(recorded_jnt_pos):
                robot.set_joint_cmd(recorded_jnt_pos[play_idx])
                play_idx += 1
            else:
                state = "idle"
                print("Playback finished.")
        else:
            robot.set_joint_cmd(obs["jnt_pos"])
        
        print("State:", state)
        node.sleep()
    
    node.spin_thread_finish()
