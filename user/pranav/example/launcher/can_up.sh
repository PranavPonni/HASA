#!/bin/bash

for IF in $(ip -o link show type can | awk -F': ' '{print $2}'); do
  echo "Configuring CAN interface: $IF"
  sudo ip link set dev "$IF" type can bitrate 1000000 restart-ms 100

  echo "Bringing up CAN interface: $IF"
  sudo ip link set dev "$IF" up
done
