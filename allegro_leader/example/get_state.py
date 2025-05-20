from allegro_leader import DynamixelDriver

def main():
    ids = [0,1,2,3,12,13,14,15]
    types = ["XL330_M077_T"] * len(ids)
    driver = DynamixelDriver(ids, types)
    driver.set_current_based_position_control_mode()
    driver.set_torque_mode(True)
    try:
        while True:
            p,v,c,t = driver.get_states()
            print("position: ",p)
            print("velocity: ",v)
            print("current:", c)
            print("torque: ", t)
    except KeyboardInterrupt:
        driver.set_torque_mode(False)
        driver.close()

if __name__ == "__main__":
    main()
