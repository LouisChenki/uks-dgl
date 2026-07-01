#!/bin/bash
# ==============================================================================
# UKS-DGL 真实世界数据实验与 HPO 远程部署启动脚本 (Remote Real Data Tuning Deploy Script)
# ==============================================================================

# 配置远程服务器连接参数
HOST="connect.westc.seetacloud.com"
PORT="44716"
USER="root"
PASS="1MfWNChRCUUv"
REMOTE_DIR="/root/uks-dgl-run"
ARCHIVE_NAME="uks_dgl_real_deploy.tar.gz"

echo "=== [1/4] 本地打包真实数据实验环境... ==="
# 打包时包含已被预处理好的 processed npz 数据，排除模拟数据以减轻包体
tar --exclude='.venv' \
    --exclude='.git' \
    --exclude='results' \
    --exclude='results_real' \
    --exclude='*.tar.gz' \
    --exclude='scratch/plots' \
    --exclude='DKNN_repo' \
    --exclude='data/synthetic_*' \
    -czf ${ARCHIVE_NAME} src data scratch

echo "--> 本地打包成功: ${ARCHIVE_NAME}"

echo "=== [2/4] 正在上传打包文件至云端服务器... ==="
expect -c "
    set timeout -1
    spawn scp -P ${PORT} ${ARCHIVE_NAME} ${USER}@${HOST}:/root/
    expect {
        \"*yes/no*\" { send \"yes\r\"; exp_continue }
        \"*password:*\" { send \"${PASS}\r\" }
    }
    expect eof
"

echo "=== [3/4] 正在配置云端服务器并启动真实数据 HPO 管线... ==="
# 在云端解压并后台启动 run_real_tuning.py，输出重定向至 real_tuning.log
expect -c "
    set timeout 60
    spawn ssh -p ${PORT} ${USER}@${HOST}
    expect {
        \"*yes/no*\" { send \"yes\r\"; exp_continue }
        \"*password:*\" { send \"${PASS}\r\" }
    }
    expect \"*#*\"
    send \"mkdir -p ${REMOTE_DIR} && mv /root/${ARCHIVE_NAME} ${REMOTE_DIR}/ && cd ${REMOTE_DIR}\r\"
    expect \"*#*\"
    send \"pkill -f 'run_real_tuning'\r\"
    expect \"*#*\"
    send \"tar -xzf ${ARCHIVE_NAME}\r\"
    expect \"*#*\"
    send \"pip install optuna scipy matplotlib pandas\r\"
    expect \"*#*\"
    send \"echo '=== 云端就绪 后台启动真实数据调参运行... ==='\r\"
    expect \"*#*\"
    send \"nohup /root/miniconda3/bin/python3 -u scratch/run_real_tuning.py > real_tuning.log 2>&1 &\r\"
    expect \"*#*\"
    send \"exit\r\"
    expect eof
"

# 清理本地临时打包
rm -f ${ARCHIVE_NAME}
echo "=== [4/4] 远程 HPO 后台挂载进程已就绪！ ==="
echo "您可以在终端执行以下指令监控进度 ( Tailing HPO logs ):"
echo "  ssh -p ${PORT} ${USER}@${HOST} (密码: ${PASS})"
echo "  tail -f ${REMOTE_DIR}/real_tuning.log"
echo "---"
