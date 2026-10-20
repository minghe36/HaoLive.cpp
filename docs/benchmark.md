# Bench 基准测试

本机已跑过的 CPU/Metal 数据见 [Bench 记录](benchmark-results.md)。

脚本：`bench/bench.py`，仅使用 Python 标准库。使用同一个 server 二进制，顺序遍历
`模型 × backend × 音频 × 模式`，每个模型/backend 单独启动进程，测试后自动退出。
每个音频/模式分别 warmup，warmup 原始记录保留，但不计入汇总。

## 运行

从仓库根目录执行，音频由调用者提供：

```sh
python3 bench/bench.py --server ./build/bin/haolive-server \
  --model ./models/x-asr-zh-en-q8_0.gguf \
  --model ./models/confucius4-r2t2-q4_k.gguf \
  --model ./models/nemotron-3.5-asr-streaming-0.6b-q8_0.gguf \
  --backend cpu --backend metal \
  --audio ./speech-zh.wav --audio ./speech-en.wav \
  --seconds 5 --iterations 5 --warmup 1 \
  --threads 4 --step-ms 800 --chunk-ms 160 \
  --stream-mode paced --output ./build/bench-paced-001
```

Moonshine Medium 当前为英文 checkpoint；测它时使用英文音频并添加
`--model ./models/moonshine-streaming-medium-q4_k.gguf`。词表内嵌，无额外文件。

- `--mode both|offline|stream`，默认 `both`。
- `--stream-mode paced`：默认。模拟采集时钟，每块音频在其末尾采样时刻到达后才发送。
  如果推理较慢，下一块会排队，记录积压，不丢弃音频。
- `--stream-mode fast`：尽快提交连续音频，不等待采集时钟，用于测量回放吞吐。
- `--seconds`：选择 WAV 开头的指定秒数。省略则使用完整文件，必须在 30 秒以内。
- `--iterations` / `--warmup`：正式运行与预热次数，默认 3 / 1。
- `--threads` / `--step-ms` / `--max-tokens`：传给 server 的推理设置。
- `--hash-models`：测试前计算权重 SHA-256，便于确认输入一致；读取权重会预热文件缓存。
- `--allow-empty`：仅用于有意测量静音。默认空 final 会使该配置失败，防止空输出产生
  看似优异的速度结果。识别正文和每块 partial 都保留在 JSON 中，便于人工检查。
- `--timeout` / `--startup-timeout`：单次请求与启动等待上限，默认 120 / 60 秒。

`--output` 指定一个尚不存在的目录。模型和音频路径均来自参数，无本地机器路径依赖。
CPU/Metal 是本次 Mac 的验收范围，脚本也接受其他 backend 名称供对应平台使用；
指定的 backend 不可用时记录错误并返回非零状态，不更换 backend。

## 指标定义

所有计时使用单调高精度时钟。请求使用直连 loopback 和 HTTP keep-alive。

| JSON/CSV 指标 | 含义 |
| --- | --- |
| `startup_ready_s` | 创建 server 进程到首次成功 `/health`，包含设备初始化、模型加载、编码图准备和 server 启动；含约 20 ms 轮询粒度。每个模型/backend 记录一次 |
| `wall_s` | 离线为完整转录 HTTP 调用耗时；流式为会话创建后，到 final 响应的总时间 |
| `rtf` / `rtf_mean` | `wall_s / 音频秒数`；离线与 fast 模式小于 1 表示本次处理快于音频时长 |
| `realtime_speed_mean` | `1 / rtf_mean` |
| `request_work_rtf` | 音频及 final 请求的耗时累计 / 音频秒数；排除模拟采集等待和创建会话的耗时，仍包含 HTTP、数据传输与 JSON 开销 |
| `session_create_s` | 流式创建会话请求耗时，保留在 JSON 原始记录中 |
| `first_text_s` | 从模拟采集开始，到首次非空文本响应；若只有 final 有正文，则使用 final 到达时间 |
| `first_partial_s` | 首次非空 partial 的到达时间；没有则为 `null` |
| `first_text_audio_s` | 首次非空文本出现时，已提交的音频秒数 |
| `final_request_s` | 显式空音频 final 请求的往返耗时 |
| `capture_to_final_s` | paced 模式下，从最后一个输入采样应到达的时刻到 final 响应，含积压；fast 模式为 `null` |
| `max_schedule_lag_s` | paced 模式中，实际发送与模拟音频就绪时刻的最大差值；fast 模式为 `null` |
| `request_p50_s` / `request_p95_s` / `request_max_s` | 正式运行的音频请求及 final 请求延迟，包含未触发推理的音频分块请求 |
| `deterministic_text` | 同一配置各次正式运行的 final 文本是否完全一致 |

首字指标测量的是首次非空识别响应，server 当前按模型推理步骤返回完整假设，
不测量解码器内部首 token 时间。paced 模式的 `wall_s` 包含采集等待，RTF 通常大于等于 1；
吞吐比较可使用离线、fast 模式或明确标注的 `request_work_rtf`。后者也不是纯计算内核耗时。
首次文本以文件起点计时，包含样本前导静音，不等于实际说话起点到文字的延迟。

p50/p95 对样本排序后线性插值。重复次数较少时，p95 不能代表稳定的尾延迟。
推荐为性能对比使用相同音频、chunk/step、线程数和 warmup，增加正式运行次数，
并保持其他机器负载接近。默认不清理操作系统文件缓存或 Metal pipeline cache，
启动时间不应称为冷启动模型加载时间。

## 输出

- `results.json`：schema_version、系统/Python 信息、二进制 SHA-256、模型路径/大小/mtime、
  可选权重 SHA-256、实际 PCM SHA-256、完整设置、启动记录、warmup/正式运行原始数据、
  每块文本/时序、汇总和错误。
- `summary.csv`：每个模型/backend/音频/模式一行，UTF-8 BOM，便于电子表格打开。
- `server-*.log`：对应进程的模型/GPU 初始化和错误日志。

每个测试项完成后更新报告；中断时保留已完成的结果并退出 130。配置失败退出非零状态，
错误运行不计入统计。输出中会包含调用者传入文件的绝对路径，分享报告前可按需脱敏。
该脚本测量单客户端端到端性能，不测多客户端并发吞吐，也不计算 WER/CER。

## 脚本回归测试

```sh
python3 -m unittest discover -s tests -p test_bench.py -v
ctest --test-dir build --output-on-failure
```

覆盖 RTF、采集时钟、推理积压、尾块保留、首次 partial/final 区分、会话取消、
warmup 排除、百分位统计、音频校验和 JSON/CSV 输出。

## Independent VAD latency

VAD uses `tests/vad_integration.py`, which warms once and measures 0.5/1/5 s
windows using the independent `/v1/vad` endpoint. It does not load ASR unless
`--asr-model` is supplied. Usage is in the README; host measurements and
model/binary identity are recorded in [VAD acceptance and latency](vad.md).


## 完整 MP3 语料、内存与质量预估

`bench/corpus_bench.py` 用于 `bench/case` 这种超过 30 秒的完整音频目录。
它只依赖 Python 标准库及 macOS 系统 `afconvert`，不会给 server 添加音频解码依赖。
每个文件转换一次，所有模型/backend 共用相同 PCM；按 20 秒无重叠窗口覆盖完整文件。
每项独立 server 进程，预热一窗、正式完整运行一轮，使用 `os.wait4` 获取该进程生命周期
峰值 RSS，避免误用所有子进程累计的 getrusage(CHILDREN)。加载、预热、推理和进程总耗时分别保留。

```sh
python3 bench/corpus_bench.py --server build-single/bin/haolive-server \
  --model models/x-asr-zh-en-q8_0.gguf \
  --model models/confucius4-r2t2-q4_k.gguf \
  --model models/moonshine-streaming-medium-q4_k.gguf \
  --model models/nemotron-3.5-asr-streaming-0.6b-q8_0.gguf \
  --backend metal --backend cpu --input bench/case \
  --output bench/results/NEW-RUN --threads 4 --segment-seconds 20 \
  --warmup 1 --iterations 1 --timeout 180
python3 bench/corpus_report.py bench/results/NEW-RUN/results.json --report 测试报告.md
```

人工参考文稿可通过 `--reference-dir /path/to/references` 提供，文件映射为相同的相对路径和
`.txt` 扩展名，例如 `en/英文演讲.txt`。未提供的文件使用留一模型伪参考差异率，并明确标记
`pseudo-reference`，不能当成真实 WER/CER。中文投票包含 X-ASR、Confucius、Nemotron，
Moonshine 英文 checkpoint 不参与中文伪参考。CPU/Metal 使用同一份参考，优先从 Metal 取候选。
单汉字+连续英文/数字的混合 token 用作中文 WER 预估；CER 去掉空白与标点后按字符计算。
`scored-results.json` 保留逐窗口参考来源、编辑距离、分母、完整输入样本覆盖验证。
真实准确率仍需要人工校对参考文本；默认不做数字读法、简繁或语义归一化。

输出包括每个测试项的 `transcript.txt`、逐窗 `segments.jsonl`、`server.log`、`metrics.json`，
以及整次运行的原始/评分 JSON、CSV 和指定 Markdown 报告。当前报告见 [测试报告](../测试报告.md)。
`--output` 必须为新目录，避免覆盖已有结果。运行默认不会清空机器文件缓存。

```sh
python3 -m unittest discover -s tests -p 'test_corpus*.py' -v
```


发生生成上限等 HTTP 错误时，完整语料脚本会记录该窗口为空输出并继续请求后续窗口，
文件状态为 `completed_with_errors`。失败窗口仍参与端到端耗时和整文件质量评分。
不能将其算成正确识别，也不能用空输出的低耗时声称模型更快。
本次早期运行使用的逐文件遇错停止版本保存在 `bench/results/20261019-corpus/protocol/`；
补测以相同二进制、权重、PCM、线程和 token 上限，仅改变遇错继续策略。
通过 `--only-errors PRIOR/results.json` 选择需补测的配置，其他参数保持原值。

合并报告时使用独立目录，保留首轮和补测原始文件：

```sh
python3 bench/corpus_report.py bench/results/20261019-corpus/results.json \
  --retry-results bench/results/20261019-corpus-retry/results.json \
  --artifacts-dir bench/results/20261019-corpus-final \
  --report 测试报告.md
```

合并器校验二进制/权重/PCM SHA-256 与核心推理参数；拒绝将不同配置的结果混入，
也不允许补测覆盖首轮成功项。首次失败保存在合并行的 `first_attempt` 字段中。
