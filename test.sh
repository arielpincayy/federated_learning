#!/bin/bash
rm -rf logs/
mkdir -p logs

nohup python3 -u main.py true  127.0.0.1:8765 127.0.0.1:8765 > logs/server.log   2>&1 &
sleep 1
nohup python3 -u main.py false 127.0.0.1:8766 127.0.0.1:8765 > logs/client1.log  2>&1 &
nohup python3 -u main.py false 127.0.0.1:8767 127.0.0.1:8765 > logs/client2.log  2>&1 &
nohup python3 -u main.py false 127.0.0.1:8768 127.0.0.1:8765 > logs/client3.log  2>&1 &