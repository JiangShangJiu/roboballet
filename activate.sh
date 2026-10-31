# RoboBallet 远程环境激活脚本
# 用法: source /root/roboballet/activate.sh
export PATH="/root/roboballet/venv/bin:/root/roboballet/venv/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:$PATH"
export PYTHONPATH="/root"
export MUJOCO_GL="egl"
export TF_CPP_MIN_LOG_LEVEL="2"
# courier 的 C++ 库(libcourier_service_cc_proto.so)依赖 tensorflow framework
# 和 serialization_cc_proto，必须让链接器能找到它们，否则 RPC 序列化时段错误。
export LD_LIBRARY_PATH="/root/roboballet/venv/lib/python3.10/site-packages/tensorflow:/root/roboballet/venv/lib/python3.10/site-packages/courier/python:$LD_LIBRARY_PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
