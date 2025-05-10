using namespace std;

#include "allegro_node_pd.h"
#include <stdio.h>
#include "ros/ros.h"

#define RADIANS_TO_DEGREES(radians) ((radians) * (180.0 / M_PI))
#define DEGREES_TO_RADIANS(angle) ((angle) / 180.0 * M_PI)

// Default PD gains (loaded from gains_pd.yaml if available)
double k_p[DOF_JOINTS] = {
  500, 800, 900, 500,
  500, 800, 900, 500,
  500, 800, 900, 500,
  1000, 700, 600, 600
};

double k_d[DOF_JOINTS] = {
  0.01, 0.01, 0.01, 0.01,
  0.01, 0.01, 0.01, 0.01,
  0.01, 0.01, 0.01, 0.01,
  0.01, 0.01, 0.01, 0.01
};

double home_pose[DOF_JOINTS] = {
  0.0, -10.0, 45.0, 45.0,
  0.0, -10.0, 45.0, 45.0,
  5.0, -5.0, 50.0, 45.0,
  60.0, 25.0, 15.0, 45.0
};

std::string pGainParams[DOF_JOINTS] = {
  "~gains_pd/p/j00", "~gains_pd/p/j01", "~gains_pd/p/j02", "~gains_pd/p/j03",
  "~gains_pd/p/j10", "~gains_pd/p/j11", "~gains_pd/p/j12", "~gains_pd/p/j13",
  "~gains_pd/p/j20", "~gains_pd/p/j21", "~gains_pd/p/j22", "~gains_pd/p/j23",
  "~gains_pd/p/j30", "~gains_pd/p/j31", "~gains_pd/p/j32", "~gains_pd/p/j33"
};

std::string dGainParams[DOF_JOINTS] = {
  "~gains_pd/d/j00", "~gains_pd/d/j01", "~gains_pd/d/j02", "~gains_pd/d/j03",
  "~gains_pd/d/j10", "~gains_pd/d/j11", "~gains_pd/d/j12", "~gains_pd/d/j13",
  "~gains_pd/d/j20", "~gains_pd/d/j21", "~gains_pd/d/j22", "~gains_pd/d/j23",
  "~gains_pd/d/j30", "~gains_pd/d/j31", "~gains_pd/d/j32", "~gains_pd/d/j33"
};

std::string initialPosition[DOF_JOINTS] = {
  "~initial_position/j00", "~initial_position/j01", "~initial_position/j02", "~initial_position/j03",
  "~initial_position/j10", "~initial_position/j11", "~initial_position/j12", "~initial_position/j13",
  "~initial_position/j20", "~initial_position/j21", "~initial_position/j22", "~initial_position/j23",
  "~initial_position/j30", "~initial_position/j31", "~initial_position/j32", "~initial_position/j33"
};

///////////////////////////////////////////////////////////////////////////////
// Constructor / Destructor
///////////////////////////////////////////////////////////////////////////////

AllegroNodePD::AllegroNodePD() : AllegroNode() {
  control_hand_ = false;
  initController(whichHand);
  lib_cmd_sub = nh.subscribe(LIB_CMD_TOPIC, 1, &AllegroNodePD::libCmdCallback, this);
  joint_cmd_sub = nh.subscribe(DESIRED_STATE_TOPIC, 1, &AllegroNodePD::setJointCallback, this);
}

AllegroNodePD::~AllegroNodePD() {
  ROS_INFO("PD controller node is shutting down");
}

///////////////////////////////////////////////////////////////////////////////
// Callbacks
///////////////////////////////////////////////////////////////////////////////

void AllegroNodePD::libCmdCallback(const std_msgs::String::ConstPtr &msg) {
  ROS_INFO("CTRL: Heard: [%s]", msg->data.c_str());
  const std::string lib_cmd = msg->data;

  if (lib_cmd == "pdControl") {
    control_hand_ = true;
  } else if (lib_cmd == "home") {
    mutex->lock();
    for (int i = 0; i < DOF_JOINTS; i++)
      desired_joint_state.position[i] = DEGREES_TO_RADIANS(home_pose[i]);
    control_hand_ = true;
    mutex->unlock();
  } else if (lib_cmd == "off") {
    control_hand_ = false;
  } else if (lib_cmd == "save") {
    mutex->lock();
    for (int i = 0; i < DOF_JOINTS; i++)
      desired_joint_state.position[i] = current_position[i];
    mutex->unlock();
  }
}

void AllegroNodePD::setJointCallback(const sensor_msgs::JointState &msg) {
  ROS_WARN_COND(!control_hand_, "Setting control_hand_ to True because of received JointState message");
  mutex->lock();
  if (msg.position.size() >= DOF_JOINTS) {
    for (int i = 0; i < DOF_JOINTS; i++) {
      desired_joint_state.position[i] = msg.position[i];
    }
  } else {
    ROS_WARN("Received JointState message does not contain enough joint positions. Expected %d, got %ld",
             DOF_JOINTS, msg.position.size());
  }
  mutex->unlock();
  control_hand_ = true;
}

///////////////////////////////////////////////////////////////////////////////
// Control Computation
///////////////////////////////////////////////////////////////////////////////

void AllegroNodePD::computeDesiredTorque() {
  if (!control_hand_) {
    for (int i = 0; i < DOF_JOINTS; i++)
      desired_torque[i] = 0.0;
    return;
  }

  // If both positions and torques are provided, issue a warning and exit.
  if (desired_joint_state.position.size() > 0 && desired_joint_state.effort.size() > 0) {
    ROS_WARN("Error: both positions and torques are specified in the desired state. You cannot control both at the same time.");
    return;
  }

  mutex->lock();
  if (desired_joint_state.position.size() == DOF_JOINTS) {
    for (int i = 0; i < DOF_JOINTS; i++) {
        double error = desired_joint_state.position[i] - current_position_filtered[i];
        desired_torque[i] = (1.0 / canDevice->torqueConversion()) *
                            (k_p[i] * error - k_d[i] * current_velocity_filtered[i]);
    }
  } else if (desired_joint_state.effort.size() > 0) {
    for (int i = 0; i < DOF_JOINTS; i++)
      desired_torque[i] = desired_joint_state.effort[i];
  }
  mutex->unlock();
}

///////////////////////////////////////////////////////////////////////////////
// Controller Initialization and Main Loop
///////////////////////////////////////////////////////////////////////////////

void AllegroNodePD::initController(const std::string &whichHand) {
  if (ros::param::has("~gains_pd")) {
    ROS_INFO("CTRL: PD gains loaded from param server.");
    for (int i = 0; i < DOF_JOINTS; i++) {
      ros::param::get(pGainParams[i], k_p[i]);
      ros::param::get(dGainParams[i], k_d[i]);
    }
  } else {
    ROS_WARN("CTRL: PD gains not loaded. Check launch file for /parameters/gains_pd.yaml. Loading default PD gains...");
  }

  if (ros::param::has("~initial_position")) {
    ROS_INFO("CTRL: Initial Pose loaded from param server.");
    double tmp;
    mutex->lock();
    desired_joint_state.position.resize(DOF_JOINTS);
    for (int i = 0; i < DOF_JOINTS; i++) {
      ros::param::get(initialPosition[i], tmp);
      desired_joint_state.position[i] = DEGREES_TO_RADIANS(tmp);
    }
    mutex->unlock();
  } else {
    ROS_WARN("CTRL: Initial position not loaded. Check launch file for /parameters/initial_position.yaml. Loading Home position instead...");
    mutex->lock();
    desired_joint_state.position.resize(DOF_JOINTS);
    for (int i = 0; i < DOF_JOINTS; i++)
      desired_joint_state.position[i] = DEGREES_TO_RADIANS(home_pose[i]);
    mutex->unlock();
  }
  control_hand_ = false;

  printf("*************************************\n");
  printf("      Joint PD Control Method        \n");
  printf("-------------------------------------\n");
  printf("  Only 'H', 'O', 'S', 'Space' works. \n");
  printf("*************************************\n");
}

void AllegroNodePD::doIt(bool polling) {
  if (polling) {
    ROS_INFO("Polling = true.");
    while (ros::ok()) {
      updateController();
      ros::spinOnce();
    }
  } else {
    ROS_INFO("Polling = false.");
    ros::Timer timer = startTimerCallback();
    ros::spin();
  }
}

///////////////////////////////////////////////////////////////////////////////
// Main Function
///////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
  ros::init(argc, argv, "allegro_hand_core_pd");
  AllegroNodePD allegroNode;
  bool polling = false;
  if (argv[1] == std::string("true"))
    polling = true;
  allegroNode.doIt(polling);
}
