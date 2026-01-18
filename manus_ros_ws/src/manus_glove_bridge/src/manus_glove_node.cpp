#include <ros/ros.h>
#include <sensor_msgs/JointState.h>

#include "ManusSDK.h"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "manus_glove_node");
    ros::NodeHandle nh;

    ros::Publisher joint_pub =
        nh.advertise<sensor_msgs::JointState>("joint_states", 10);

    // --- Initialize MANUS Integrated ---
    CoreSdk_InitializeIntegrated();

    while (ros::ok())
    {
        // Get skeleton data from MANUS
        // (Pseudo-code: exact calls depend on SDK headers)

        sensor_msgs::JointState js;
        js.header.stamp = ros::Time::now();

        // Example (you will map real MANUS joints)
        js.name.push_back("index_mcp");
        js.position.push_back(0.5); // radians

        joint_pub.publish(js);

        ros::spinOnce();
    }

    CoreSdk_ShutDown();
    return 0;
}
