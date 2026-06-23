#!/bin/bash
# ==============================================================================
# UKS-DGL 远程云端实验状态监控脚本 (Remote Experiment Monitor Script)
# ==============================================================================

HOST="connect.westd.seetacloud.com"
PORT="33493"
USER="root"
PASS="Eadv0a4/xeMF"
REMOTE_DIR="/root/uks-dgl-run"

echo "=== 正在连接云端服务器查询进度... ==="
expect -c "
    set timeout 20
    spawn ssh -p ${PORT} ${USER}@${HOST}
    expect {
        \"*yes/no*\" { send \"yes\r\"; exp_continue }
        \"*password:*\" { send \"${PASS}\r\" }
    }
    expect \"*#*\"
    send \"echo '--- 进程状态 ---' && ps aux | grep run_experiment.py | grep -v grep\r\"
    expect \"*#*\"
    send \"echo '--- 实时日志尾部 ---' && tail -n 25 ${REMOTE_DIR}/train.log\r\"
    expect \"*#*\"
    send \"exit\r\"
    expect eof
"
