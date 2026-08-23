export MODEL_PATH=/shared/amdgpu/home/fan_wu2_7kq/models/DeepSeek-V4-Pro
export INFMAX_CONTAINER_WORKSPACE=/shared/amdgpu/home/fan_wu2_7kq/semi/InferenceX
local/run_local.sh dsv4-fp4-mi355x-sglang-agentic-conc32 \
  --config local/recipes/dsv4-fp4-mi355x-sglang-agentic-conc32.yaml
