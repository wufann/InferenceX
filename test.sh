#export MODEL_PATH=/shared/amdgpu/home/fan_wu2_7kq/models/DeepSeek-V4-Pro
#export INFMAX_CONTAINER_WORKSPACE=/shared/amdgpu/home/fan_wu2_7kq/semi/InferenceX
## local/run_local.sh dsv4-fp4-mi355x-sglang-agentic-conc32 \
##   --config local/recipes/dsv4-fp4-mi355x-sglang-agentic-conc32.yaml
#local/run_local.sh dsv4-fp4-mi355x-sglang-agentic-conc48 \
#    --config local/recipes/dsv4-fp4-mi355x-sglang-agentic-conc48.yaml

# bash local/aggregate_agentic.sh /workspace/results/dsv4_tp8_conc48_kvdram-hicache_spec-mtp_fp4_sglang_local/



#### nv B300
#local/run_local.sh dsv4-fp4-b300-sglang-agentic-conc384 \
#    --config local/recipes/dsv4-fp4-b300-sglang-agentic-conc384.yaml

# RUNNER_TYPE=b300 local/aggregate_agentic.sh /workspace/results/dsv4_tp8_conc384_*_local
# 
# python3 local/show_agentic.py /workspace/results/dsv4_tp8_conc384_*/dsv4_tp8_conc384_*.json
#

export MODEL_PATH=/shared/amdgpu/home/fan_wu2_7kq/models/Qwen3.8-Flash-Next-MXFP4-PLEFP8
local/run_local.sh qwen3.8next-fp4-mi355x-sglang-agentic-conc8-12-16 \
    --config local/recipes/qwen3.8next-fp4-mi355x-sglang-agentic-conc8-12-16.yaml
    #--min-conc 16 --max-conc 16
