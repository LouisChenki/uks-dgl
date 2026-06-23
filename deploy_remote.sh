#!/bin/bash
# ==============================================================================
# UKS-DGL 远程云端服务器一键部署与运行控制脚本 (Remote Deployment Control Script)
# ==============================================================================

# 配置参数 (Configuration Parameters)
HOST="connect.westd.seetacloud.com"
PORT="33493"
USER="root"
PASS="Eadv0a4/xeMF"
REMOTE_DIR="/root/uks-dgl-run"
ARCHIVE_NAME="uks_dgl_deploy.tar.gz"

echo "=== [1/5] 本地项目打包隔离中... ==="
# 排除虚拟环境与以往的实验结果目录
tar --exclude='.venv' \
    --exclude='.git' \
    --exclude='results_*' \
    --exclude='results' \
    --exclude='*.tar.gz' \
    -czf ${ARCHIVE_NAME} src data requirements.txt run_experiment.py

echo "--> 本地项目打包成功: ${ARCHIVE_NAME}"

# 使用 macOS 自带的 expect 实现免交互输入密码拷贝
echo "=== [2/5] 正在通过 SCP 上传打包文件至云端服务器... ==="
expect -c "
    set timeout -1
    spawn scp -P ${PORT} ${ARCHIVE_NAME} ${USER}@${HOST}:/root/
    expect {
        \"*yes/no*\" { send \"yes\r\"; exp_continue }
        \"*password:*\" { send \"${PASS}\r\" }
    }
    expect eof
"

echo "=== [3/5] 正在配置云端服务器运行环境并启动实验... ==="
# 登录云端服务器解压，并启动实验 (使用 nohup 后台静默挂载，实时将日志 flush 输出至 train.log)
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
    send \"tar -xzf ${ARCHIVE_NAME}\r\"
    expect \"*#*\"
    send \"pip install scipy matplotlib\r\"
    expect \"*#*\"
    send \"echo '=== 云端环境就绪 正在后台启动 AutoML 寻优实验... ==='\r\"
    expect \"*#*\"
    send \"nohup python3 -u run_experiment.py > train.log 2>&1 &\r\"
    expect \"*#*\"
    send \"exit\r\"
    expect eof
"

echo "=== [4/5] 云端后台进程启动成功！ ==="
echo "您可以通过以下指令登录云端监控实时训练日志 (AutoML Training Logs):"
echo "  ssh -p ${PORT} ${USER}@${HOST}  (密码: ${PASS})"
echo "  tail -f ${REMOTE_DIR}/train.log"

# 删除本地临时压缩包
rm -f ${ARCHIVE_NAME}
echo "=== [5/5] 本地临时文件清理完毕，等待实验收官。 ==="
