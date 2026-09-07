# voice_hub

自建「小智式」语音服务器：电脑跑大脑，Jetson 只负责麦/喇叭。**每种机械臂自己带着 JSON 格式和校验**，服务器不写死某一种控制协议。

交接（已完成 / 下一步 / 终极目标）：见 [docs/交接文档.md](docs/交接文档.md)。

## ASR / TTS / LLM

| 模块 | 现在 | 以后换成在线 |
|---|---|---|
| LLM | 阿里云百炼 Qwen `https://dashscope.aliyuncs.com/compatible-mode/v1`，请求里关闭思考 | 模型名自己填，例如 `qwen-flash` |
| ASR | 豆包流式 `ASR_PROVIDER=doubao`（火山 SAUC WebSocket） | 本机 Paraformer：`ASR_PROVIDER=sherpa_local` |
| TTS | 豆包语音 `TTS_PROVIDER=doubao`（火山 HTTP 单向按块下发，设备端边收边播） | 双向 WebSocket / 本机 Kokoro |
| 闲聊 | 豆包端到端实时语音 `DIALOG_APPID` + `DIALOG_ACCESS_TOKEN`（听+想+说一条 WebSocket） | 不填则回退百炼 + 豆包 TTS |
| 记忆 | 按机械臂记住城市等，问天气不再问在哪 | `~/.local/share/voice_hub/memory/` |
| 搜索 | 秘塔；天气无城市先问城市；查询带真实日期 | 不走火山内置联网 |

Jetson **不要拷** Kokoro / Paraformer / Dify。模型只在电脑上一份，路径写在 `env.example`。

## 目录

- `server/` 电脑进程：WebSocket、适配器、ASR/TTS/LLM 插件
- `adapters/<robot_id>/` 给 LLM 看的知识 + 口语规则（可生成该臂 JSON 建议）
- `jetson_client/` 薄客户端，不含任何一种臂的字段名
- `lyz/src/tianlian_voice_bridge/` 天链双臂的 schema / 校验 / ROS 发布（跟机械臂走）
- `bridges/demo_single.py` 另一套 JSON，用来证明两台臂互不串

## 本机空转（不需要真实机械臂）

```bash
cd /home/ning/model/voice_hub
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
# 无模型也可测文本路径：
export ASR_PROVIDER=none TTS_PROVIDER=dummy
python3 -m unittest discover -s tests -q

# 启动服务器前编辑 .env，填百炼 Key 和模型名：
#   nano /home/ning/model/voice_hub/.env
#   CHAT_LLM_API_KEY=（https://bailian.console.aliyun.com/?tab=model#/api-key）
#   CHAT_LLM_MODEL=qwen-flash
set -a && source .env && set +a
export TTS_PROVIDER=dummy   # 没装 sherpa 时
python3 -m server
# 另开终端：
PYTHONPATH="/home/ning/model/voice_hub:/home/ning/model/lyz/src/tianlian_voice_bridge" \
  python3 jetson_client/voice_client.py --text "你有几个关节"
```

`VOICE_ENABLE_ROBOT_COMMAND` 默认关闭。真动天链前：本机校验通过 + 急停就绪后再设为 1。

## 其它 Linux 设备怎么连（设备端）

主机先启动：

```bash
cd /home/ning/model/voice_hub
bash scripts/run_server.sh
```

在 Jetson 或另一台 Linux 上（**不要拷主机 `.env`**）：

```bash
# 至少需要 device_ui/ jetson_client/ bridges/ requirements.txt
pip install -r requirements.txt
bash scripts/run_device_ui.sh
# 可选：安装桌面图标
bash scripts/install_linux_device_desktop.sh
```

浏览器打开 `http://127.0.0.1:8780`。主机地址填局域网 IP，例如：

```text
ws://10.101.67.111:8765
```

设备和主机须同一 Wi‑Fi / 同一网段（或 Tailscale）。不要填 `127.0.0.1`（那是这台设备自己），不要填 `198.18.x`（Clash 假 IP），不要做 8765 公网端口转发。换设备时拷这份程序，只改主机 IP。

