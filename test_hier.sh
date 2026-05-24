#!/bin/bash
rm -rf logs/
mkdir -p logs

nohup python3 -u hierarchy.py null            127.0.0.1:8765 2 > logs/raiz_8765.log  2>&1 &
sleep 1
nohup python3 -u hierarchy.py 127.0.0.1:8765 127.0.0.1:8766 2 > logs/inter_8766.log 2>&1 &
nohup python3 -u hierarchy.py 127.0.0.1:8765 127.0.0.1:8767 2 > logs/inter_8767.log 2>&1 &
sleep 1
nohup python3 -u hierarchy.py 127.0.0.1:8766 127.0.0.1:8768 0 > logs/hoja_8768.log  2>&1 &
nohup python3 -u hierarchy.py 127.0.0.1:8766 127.0.0.1:8769 0 > logs/hoja_8769.log  2>&1 &
nohup python3 -u hierarchy.py 127.0.0.1:8767 127.0.0.1:8770 0 > logs/hoja_8770.log  2>&1 &
nohup python3 -u hierarchy.py 127.0.0.1:8767 127.0.0.1:8771 0 > logs/hoja_8771.log  2>&1 &