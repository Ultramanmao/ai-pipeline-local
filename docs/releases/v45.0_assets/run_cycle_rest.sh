#!/bin/bash
# 转场03 四季循环补全 — 分阶段运行脚本（GPU 空闲时由 Hermes 驱动）
# 用法: bash run_cycle_rest.sh anchors | videos | concat
#
# 阶段依赖（顺序不可反）：
#   anchors  → 需要 ComfyUI 在线（8188）；出早春锚 D3_tr03c_spring_00001_.png
#   videos   → 需要 ComfyUI **下线**（RAM 杀机，两次实测翻车）；跑 seg3/seg4 两段 FreeVideo
#              可选: DO_SEG2_FIX=1 先重烧 seg2（草枯修复，白名单外需用户点头才开）
#   concat   → 纯 CPU；拼四季全循环 + 抽帧 QC 素材
#
# 守窗/kill ComfyUI 不在本脚本内（破坏性动作走守窗协议+用户批准，Hermes 手动执行）。

set -u
cd "$(dirname "$0")"
FV="E:/WSL/ComfyUI/custom_nodes/FreeVideo/freevideo.cmd"
CACHE="E:/WSL/FreeVideo/prepared/edge-da37681f5a1f7015/int8-convrot/cache"
LORA_PATH="E:/WSL/ComfyUI/models/loras/wushu_spatial_physics_clean_3000_pruned.safetensors"

run_fv () {  # run_fv <prompt_file> <media_json> <out_mp4> <seed>
  "$FV" --root E:/WSL/FreeVideo generate --task fl2va \
    --prompt-file "$1" --media "$2" \
    --width 1344 --height 768 --frames 90 --seed "$4" \
    --cache "$CACHE" --out "$3"
}

case "${1:-}" in
  anchors)
    echo "=== stage: anchors (needs ComfyUI UP) ==="
    curl -s --max-time 5 http://127.0.0.1:8188/queue > /dev/null || { echo "ComfyUI 不在线，先拉起再跑本阶段"; exit 1; }
    uv run python gen_anchors3.py
    ;;

  videos)
    echo "=== stage: videos (needs ComfyUI DOWN) ==="
    if netstat -ano | grep 'LISTENING' | grep -q ':8188'; then
      echo "8188 仍有监听 — ComfyUI 未下线，拒绝起跑（RAM 杀机）"; exit 1
    fi
    if [ "${DO_SEG2_FIX:-0}" = "1" ]; then
      echo "--- seg2 fix (草枯修复) ---"
      run_fv video_prompt_seg2_fix.txt media_seg2.json tr03b_seg2_fix_grass.mp4 20261018 \
        > seg2fix.log 2>&1
    fi
    echo "--- seg3: winter -> spring ---"
    run_fv video_prompt_seg3.txt media_seg3.json tr03c_seg3_winter_spring.mp4 20261019 \
      > seg3.log 2>&1
    echo "--- seg4: spring -> summer (loop close) ---"
    run_fv video_prompt_seg4.txt media_seg4.json tr03c_seg4_spring_summer.mp4 20261020 \
      > seg4.log 2>&1
    echo "videos done"; ls -la tr03c_seg*.mp4
    ;;

  concat)
    echo "=== stage: concat + QC frames ==="
    SEG2="tr03b_seg2_autumn_winter.mp4"
    [ -f tr03b_seg2_fix_grass.mp4 ] && SEG2="tr03b_seg2_fix_grass.mp4" && echo "(使用 seg2 修复版)"
    printf "file 'tr03b_seg1_summer_autumn.mp4'\nfile '%s'\nfile 'tr03c_seg3_winter_spring.mp4'\nfile 'tr03c_seg4_spring_summer.mp4'\n" "$SEG2" > concat_cycle.txt
    ffmpeg -v quiet -f concat -safe 0 -i concat_cycle.txt \
      -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p tr03c_full_cycle.mp4 -y
    ffprobe -v quiet -show_entries format=duration -of csv=p=0 tr03c_full_cycle.mp4
    mkdir -p qc
    for s in tr03c_seg3_winter_spring tr03c_seg4_spring_summer; do
      ffmpeg -v quiet -i "$s.mp4" -vf "select='eq(n,4)+eq(n,25)+eq(n,47)+eq(n,66)+eq(n,80)+eq(n,89)',scale=640:-1,tile=3x2" -frames:v 1 -q:v 2 "qc/${s}_sheet.jpg" -y
      uv run --with pillow python C:/hermes/skills/mlops/freevideo-h3-pipeline/scripts/tail_freeze_qc.py "$s.mp4" | tail -2
    done
    ;;

  *)
    echo "用法: bash run_cycle_rest.sh anchors | videos | concat"; exit 1
    ;;
esac
