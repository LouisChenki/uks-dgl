#!/bin/bash
# ==============================================================================
# UKS-DGL 远程云端结果拉取与同步脚本 (Remote Results Retrieval Script)
# ==============================================================================

HOST="connect.westd.seetacloud.com"
PORT="33493"
USER="root"
PASS="Eadv0a4/xeMF"
REMOTE_DIR="/root/uks-dgl-run/results_20260608_run11"
LOCAL_DIR="results_20260608_run11"

echo "=== [1/2] 正在建立加密通道拉取云端结果数据... ==="
# 创建本地存放目录
mkdir -p "${LOCAL_DIR}"

expect -c "
    set timeout -1
    spawn scp -r -P ${PORT} ${USER}@${HOST}:${REMOTE_DIR}/* ${LOCAL_DIR}/
    expect {
        \"*yes/no*\" { send \"yes\r\"; exp_continue }
        \"*password:*\" { send \"${PASS}\r\" }
    }
    expect eof
"

echo "=== [2/2] 云端实验结果同步成功！ ==="
echo "本地结果存放于: ${LOCAL_DIR}/"
