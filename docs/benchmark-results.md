# 本机 Bench 记录

本次用于验证 bench 脚本与记录本机基线。机器报告为 macOS 26.6.2 / arm64，Python 3.14.6；Metal 日志标识 GPU 为 Apple M4 Max。

使用中英文 WAV 各自开头的 5 秒，4 线程，模型 step 800 ms，上传 chunk 160 ms，顺序运行。权重已在启动前计算 SHA-256，文件缓存和 Metal pipeline cache 未清理。

## Fast 回放与离线

每个音频/模式 warmup 1 次，正式测量 3 次。共 16 项配置、64 次运行，其中正式运行 48 次。所有 final 非空，同一配置正式运行的文本一致。

| 模型 | Backend | 音频 | 离线平均 RTF | Fast 流式平均 RTF |
| --- | --- | --- | --- | --- |
| x-asr-zh-en-q8_0 | cpu | 英文 | 0.0698 | 0.0730 |
| x-asr-zh-en-q8_0 | cpu | 中文 | 0.0662 | 0.0696 |
| x-asr-zh-en-q8_0 | metal | 英文 | 0.0926 | 0.0884 |
| x-asr-zh-en-q8_0 | metal | 中文 | 0.0900 | 0.0929 |
| confucius4-r2t2-q4_k | cpu | 英文 | 0.1967 | 0.9084 |
| confucius4-r2t2-q4_k | cpu | 中文 | 0.2029 | 0.9077 |
| confucius4-r2t2-q4_k | metal | 英文 | 0.0336 | 0.1394 |
| confucius4-r2t2-q4_k | metal | 中文 | 0.0346 | 0.1436 |

RTF = 处理时间 / 音频时长，小于 1 表示处理快于音频长度。

## Paced 模拟实时输入

Confucius、同一中文 5 秒音频，每项 warmup 1 次、正式运行 2 次。

| Backend | 首次非空 partial p50 | 首字对应已提交音频 | 采集结束到 final p50 | 请求工作时间 RTF |
| --- | --- | --- | --- | --- |
| cpu | 2.009 s | 1.600 s | 1.726 s | 0.9354 |
| metal | 1.691 s | 1.600 s | 0.135 s | 0.1621 |

首次文本从文件起点计时，包含音频内容和模拟采集等待。请求工作时间 RTF 排除采集等待，仍包含 HTTP/JSON 开销。paced 的 wall RTF 不用于吞吐比较。

## 输入与产物身份

- server SHA-256：`794b8f4b9ca93c23023eca33de794b70730e83ad3ad9c4a8e93197adf07050b2`。
- `x-asr-zh-en-q8_0.gguf` SHA-256：`1ca120084a1517cf02d96e44cdd9a9544f0c887f6d0149c9f85d151be6833a61`。
- `confucius4-r2t2-q4_k.gguf` SHA-256：`a754788730d7b2a62542de295d37e431e79795d5bea418fa0e4f9618f0f8e3b3`。
- `jfk.wav` 前 5 秒 PCM SHA-256：`329d61c2b83e8e393b0d728ef43bc6c51aa5f798def1162f2dc60894651cba95`。
- `paraformer_zh.wav` 前 5 秒 PCM SHA-256：`dde1270fc7abae5efbc91a67e16025ab28a614dd2cc4b2c1441c4f6fef77e66b`。

完整 JSON、CSV、逐块时序和日志保存在本机 `build/bench-mvp-fast/` 与 `build/bench-mvp-paced/`；这些包含调用者文件路径的原始报告未纳入 Git。

15 个 bench 单元测试与 CTest 的 3 个测试目标通过；不可用 backend 的错误报告、非零退出和子进程回收也已实测。上述少量样本不代表一般 WER/CER、冷启动性能或稳定的尾延迟分布。

## 新增模型与最终单文件程序（2026-10-19）

本节使用最终交付二进制，SHA-256：`f102153ad93efc8f9913df062f53144632266f7adc29e4b73cb62c5390149194`。
机器为同一台 Apple M4 Max，macOS arm64，Python 3.12.4。
英文样例前 5 秒，4 线程、step 800 ms、chunk 160 ms；每项预热 1 次、正式 3 次。
离线与 paced 流式共 8 项配置、32 次运行（含预热），无错误、无空 final，同配置正式运行文本一致。

| 模型 | Backend | 离线平均 RTF | 首次非空 partial p50 | 采集结束到 final p50 | 流式请求工作 RTF |
| --- | --- | --- | --- | --- | --- |
| Moonshine Medium | cpu | 0.0542 | 0.876 s | 0.350 s | 0.2606 |
| Moonshine Medium | metal | 0.0097 | 0.842 s | 0.083 s | 0.0928 |
| Nemotron 3.5 | cpu | 0.1225 | 1.693 s | 0.142 s | 0.1659 |
| Nemotron 3.5 | metal | 0.0688 | 1.708 s | 0.093 s | 0.1368 |

首字从文件起点计时，包含音频内容与采集等待；不是说话起点到文字的延迟。
paced 的 wall RTF 包含 5 秒采集等待，不能用来评价计算吞吐。
上述数据为本机小样本诊断，缓存未清理；测量期间曾执行 backend 枚举并编译 Metal pipeline，
不应视为严格隔离负载下的性能结论。

完整 JSON、CSV 与 server 日志：`build/bench-single-file/`；二进制 hash 与单文件交付包一致。
Moonshine 为本次指定的英文 checkpoint；Nemotron 的中文功能验收另见 `validation.md`。
前文旧模型基线属于此前版本，不与本节数据混称为同一二进制测量。
