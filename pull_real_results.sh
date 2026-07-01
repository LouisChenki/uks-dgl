#!/usr/bin/env zsh
# -*- coding: utf-8 -*-
# 一键自动拉取云端真实世界数据集对比实验成果脚本

REMOTE_HOST="connect.westc.seetacloud.com"
REMOTE_PORT="44716"
REMOTE_USER="root"
REMOTE_PASS="1MfWNChRCUUv"
REMOTE_DIR="/root/uks-dgl-run/results_real/"
LOCAL_DIR="./results_real"

echo "=== [1/2] 开始自动同步云端真实数据集指标与权重... ==="

expect -c "
    set timeout 120
    spawn scp -P ${REMOTE_PORT} -r ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR} ${LOCAL_DIR}_temp
    expect {
        \"*yes/no*\" { send \"yes\r\"; exp_continue }
        \"*password:*\" { send \"${REMOTE_PASS}\r\" }
    }
    expect eof
"

if [ -d "${LOCAL_DIR}_temp/results_real" ]; then
    # 规避 scp 创建多层嵌套目录，进行物理规整
    mkdir -p ${LOCAL_DIR}
    cp -r ${LOCAL_DIR}_temp/results_real/* ${LOCAL_DIR}/
    rm -rf ${LOCAL_DIR}_temp
    echo "=== [2/2] 真实数据成果已完美同步至本地 ${LOCAL_DIR}/ ！ ==="
else
    echo "--> [Warning] 同步可能未完全成功，请检查云端路径是否存在。"
    rm -rf ${LOCAL_DIR}_temp
fi
