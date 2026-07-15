## 挂载 dtfs 文件系统
mount -t dtfs  /llm_workspace_1P  /llm_workspace_1P

## 卸载 dtfs 文件系统
umount /llm_workspace_1P

# docker tag && push
docker tag quay.io/ascend/vllm-ascend:main-a3 cis-pengcheng.cmecloud.cn/ascendhub/quay.io/ascend/vllm-ascend:main-a3
docker push cis-pengcheng.cmecloud.cn/ascendhub/quay.io/ascend/vllm-ascend:main-a3

## 进入容器
docker exec -it mindspeed-llm-env /bin/bash
docker exec -it vllm-ascend-env-a3  /bin/bash
docker exec -it 2cd3a9664398 /bin/bash

# 转换模型
nohup bash scripts/ckpt_convert_hf2mcore.sh > hf2mcore.log  2>&1 &
nohup bash scripts/ckpt_convert_kimi2_hf2mcore.sh > ckpt_convert_kimi2_hf2mcore_step900_v2.log 2>&1 &
nohup bash scripts/ckpt_convert_kimi2_mcore2hf.sh >  ckpt_convert_kimi2_mcore2hf_step900_v2.log 2>&1 &
nohup bash scripts/ckpt_convert_mcore2hf.sh > mcore2hf.log  2>&1 &
systemctl daemon-reload && systemctl start docker

# 设置 kimi-home
export KIMI_CODE_HOME=/home/jianzhnie/llmtuner/kimi-code

# 开启代理
source /home/jianzhnie/llmtuner/llm/EasyInfer/tools/host_proxy.sh && pon -f && pstatus && echo && pverify 5

# 开启 claude 
source /home/jianzhnie/llmtuner/llm/EasyInfer/claude_env.sh

## lleval 环境
source /home/jianzhnie/llmtuner/software/miniconda3/bin/activate llmeval