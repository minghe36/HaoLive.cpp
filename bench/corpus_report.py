#!/usr/bin/env python3
"""Render measured corpus results and explicitly labelled pseudo-reference estimates."""
import argparse
import csv
import copy
import json
import statistics
import unicodedata
from pathlib import Path

COMPLETED = {"ok", "completed_with_errors"}


def normalize(text):
    return unicodedata.normalize("NFKC", text).casefold().replace("’", "'")


def tokens(text, kind):
    text = normalize(text)
    if kind == "cer":
        return [ch for ch in text if ch.isalnum()]
    result, buffer = [], []
    def flush():
        if buffer:
            value = "".join(buffer).strip("'")
            if value:
                result.append(value)
            buffer.clear()
    for ch in text:
        han = '\u3400' <= ch <= '\u9fff' or '\U00020000' <= ch <= '\U000323af'
        if han:
            flush()
            result.append(ch)
        elif ch == "'" and buffer:
            continue  # Ignore internal apostrophes without splitting contractions.
        elif ch.isalnum():
            buffer.append(ch)
        else:
            flush()
    flush()
    return result


def distance(reference, hypothesis):
    """Exact unit-cost Levenshtein distance using Myers bit vectors."""
    if not reference:
        return len(hypothesis)
    positions = {}
    for index, token in enumerate(reference):
        positions[token] = positions.get(token, 0) | (1 << index)
    mask = (1 << len(reference)) - 1
    top = 1 << (len(reference) - 1)
    positive, negative, score = mask, 0, len(reference)
    for token in hypothesis:
        equal = positions.get(token, 0)
        vertical = equal | negative
        horizontal = (((equal & positive) + positive) ^ positive) | equal
        high = negative | ~(horizontal | positive)
        low = positive & horizontal
        score += bool(high & top) - bool(low & top)
        high = (high << 1) | 1
        low <<= 1
        positive = (low | ~(vertical | high)) & mask
        negative = (high & vertical) & mask
    return score


def medoid(candidates):
    """Choose among peers, never the evaluated architecture itself."""
    if not candidates:
        raise ValueError("no independent candidate for pseudo reference")
    sequences = {name: tokens(text, "cer") for name, text in candidates.items()}
    def cost(name):
        return sum(distance(sequences[name], other) / max(1, len(sequences[name]), len(other))
                   for peer, other in sequences.items() if peer != name)
    return min(sorted(candidates), key=lambda name: (cost(name), name))


def reference_segments(report, row, reference_dir=None):
    relative = Path(row["audio"])
    # Full human transcripts need full-file alignment rather than arbitrary
    # automatic segment boundaries; return one paired full-file observation.
    if reference_dir:
        path = reference_dir / relative.with_suffix(".txt")
        if path.is_file():
            return [(path.read_text().strip(), row["text"], {"source": str(path), "kind": "provided-reference"})]
    pool = {}
    for other in report["results"]:
        if other["status"] not in COMPLETED or other["audio"] != row["audio"] or other["model"] == row["model"]:
            continue
        if relative.parts[0] == "zh" and not other["model"].startswith(("x-asr", "confucius", "nemotron")):
            continue  # English checkpoints cannot vote on Chinese truth.
        prior = pool.get(other["model"])
        if prior is None or other["backend"] == "metal":
            pool[other["model"]] = other
    result = []
    for segment in row["segments"]:
        index = segment["index"]
        candidates = {name: other["segments"][index]["text"] for name, other in pool.items()
                      if len(other["segments"]) > index and "error" not in other["segments"][index]}
        chosen = medoid(candidates)
        result.append((candidates[chosen], segment["text"], {"kind": "leave-one-architecture-out-pseudo",
                       "model": chosen, "backend": pool[chosen]["backend"], "segment": index,
                       "start_s": segment["start_s"], "end_s": segment["end_s"],
                       "candidate_models": sorted(candidates)}))
    return result


def quality(report, row, reference_dir=None):
    pairs = reference_segments(report, row, reference_dir)
    result = {"references": [{**meta, "reference": reference} for reference, _, meta in pairs],
              "reference_kind": "provided-reference" if all(meta["kind"] == "provided-reference" for _, _, meta in pairs) else "pseudo-reference"}
    for kind in ["wer", "cer"]:
        counts = [(tokens(reference, kind), tokens(hypothesis, kind)) for reference, hypothesis, _ in pairs]
        errors = sum(distance(reference, hypothesis) for reference, hypothesis in counts)
        length = sum(len(reference) for reference, _ in counts)
        result[kind] = errors / length if length else None
        result[kind + "_edits"] = errors
        result[kind + "_reference_units"] = length
    return result


def percent(value):
    return "—" if value is None else f"{value * 100:.2f}%"


def summary(rows):
    good = [row for row in rows if row["status"] in COMPLETED]
    seconds = sum(r["audio_s"] for r in good)
    elapsed = sum(r["inference_mean_s"] for r in good)
    value = {"files": len(good), "audio_s": seconds, "inference_s": elapsed,
             "rtf": elapsed / seconds if seconds else None,
             "peak_rss_mib": max((r["peak_rss_bytes"] / 2**20 for r in good), default=0),
             "startup_mean_s": statistics.mean(r["startup_s"] for r in good) if good else None,
             "failed_segments":sum(r.get("failed_segments",0) for r in good)}
    for kind in ["wer", "cer"]:
        denominator = sum(r["quality"][kind + "_reference_units"] for r in good)
        value[kind] = sum(r["quality"][kind + "_edits"] for r in good) / denominator if denominator else None
    return value


def coverage(report):
    expected = {(backend, Path(model["path"]).name, audio["relative"])
                for backend in report["settings"]["backend"] for model in report["models"] for audio in report["audio"]}
    observed = {(row["backend"], row["model"], row["audio"]) for row in report["results"]}
    audio = {a["relative"]: a for a in report["audio"]}
    errors = []
    for row in report["results"]:
        if row["status"] not in COMPLETED:
            continue
        for run in row["runs"]:
            segments = run["segments"]
            if sum(s["samples"] for s in segments) != audio[row["audio"]]["samples"] or not segments or segments[0]["start_s"] != 0:
                errors.append([row["backend"],row["model"],row["audio"],"sample count mismatch"])
            elif any(abs(a["end_s"] - b["start_s"]) > 1e-9 for a,b in zip(segments,segments[1:])):
                errors.append([row["backend"],row["model"],row["audio"],"gap or overlap"])
    return {"expected_cases":len(expected),"observed_cases":len(report["results"]),
            "missing_cases":sorted(expected-observed),"extra_cases":sorted(observed-expected),
            "duplicate_cases":len(report["results"])-len(observed),"sample_errors":errors}


def merge_results(base, base_dir, retry_reports):
    merged=copy.deepcopy(base)
    for row in merged["results"]:
        row["artifact_dir"]=str((base_dir / row["artifact_dir"]).resolve())
    identity=lambda row:(row["backend"],row["model"],row["audio"])
    indices={identity(row):index for index,row in enumerate(merged["results"])}
    for retry,retry_dir in retry_reports:
        if retry["server"]["sha256"] != base["server"]["sha256"]:
            raise ValueError("retry used a different server")
        expected_audio={a["relative"]:a["pcm_sha256"] for a in base["audio"]}
        if any(expected_audio.get(a["relative"]) != a["pcm_sha256"] for a in retry["audio"]):
            raise ValueError("retry used different PCM")
        expected_models={Path(m["path"]).name:m["sha256"] for m in base["models"]}
        if any(expected_models.get(Path(m["path"]).name) != m["sha256"] for m in retry["models"]):
            raise ValueError("retry used different weights")
        for key in ["threads","segment_seconds","max_tokens","iterations","warmup"]:
            if retry["settings"][key] != base["settings"][key]: raise ValueError(f"retry changed {key}")
        for row in retry["results"]:
            index=indices[identity(row)]
            old=merged["results"][index]
            if old["status"] == "ok": raise ValueError("retry would replace a successful baseline")
            updated=copy.deepcopy(row)
            updated["first_attempt"]=old
            updated["artifact_dir"]=str((retry_dir / row["artifact_dir"]).resolve())
            merged["results"][index]=updated
        merged["finished_at"]=retry["finished_at"]
    return merged


def render(report, path, output, reference_dir=None):
    report["coverage"] = coverage(report)
    for row in report["results"]:
        if row["status"] in COMPLETED:
            row["quality"] = quality(report, row, reference_dir)
    (output / "scored-results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    fields = ["backend", "model", "audio", "audio_s", "inference_mean_s", "rtf", "speed", "startup_s",
              "process_elapsed_s", "peak_rss_mib", "wer_est", "cer_est", "reference_kind", "empty_segments", "failed_segments", "status"]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in report["results"]:
            q = row.get("quality", {})
            writer.writerow({**row, "peak_rss_mib": row.get("peak_rss_bytes", 0) / 2**20,
                             "wer_est": q.get("wer"), "cer_est": q.get("cer"), "reference_kind": q.get("reference_kind")})
    relative = lambda target: Path(__import__('os').path.relpath(target, path.parent)).as_posix()
    lines = ["# haolive.cpp 四模型音频 Benchmark 测试报告", "",
             f"- 开始：{report['started_at']}；结束：{report.get('finished_at', '进行中')}。",
             f"- 硬件：{report['host']['sysctl'][3]} / {int(report['host']['sysctl'][1])/2**30:.0f} GiB / {report['host']['sysctl'][2]} 核 / {report['host']['sysctl'][0]}；系统：{report['host']['platform']}。",
             f"- 输入：{len(report['audio'])} 个完整音频，共 {sum(a['audio_s'] for a in report['audio']):.3f} 秒；语种分类取自 en/zh 目录，非人工逐句鉴定。",
             f"- 无错误文件项：{sum(r['status'] == 'ok' for r in report['results'])}；包含失败窗口但已尝试完整音频的文件项：{sum(r['status'] == 'completed_with_errors' for r in report['results'])}；失败窗口总数：{sum(r.get('failed_segments',0) for r in report['results'])}。",
             f"- 正式结果：{sum(r['status'] in COMPLETED for r in report['results'])}/{report['coverage']['expected_cases']} 项已尝试完整音频；其中 {sum(r['status'] == 'ok' for r in report['results'])} 项所有正式请求无错误（不代表文本正确）。", "",
             f"- 正式窗口：共 {sum(len(r.get('segments',[])) for r in report['results'])} 个，其中 {sum('error' not in s for r in report['results'] for s in r.get('segments',[]))} 个请求无错误；不含预热及被补测替换的首次失败尝试。", "",
             "## 测试方法与指标边界", "",
             "1. 使用同一单文件 server，CPU / Metal 串行执行，不启用 VAD、不跳过静音。四模型均处理全部 10 个文件；Moonshine Medium 是英文 checkpoint，中文结果属于越界能力探测；Nemotron 3.5 是多语模型，官方范围包含普通话，本库使用 GGUF 中的 auto 语言提示。语言范围来源：[Moonshine 模型卡](https://huggingface.co/moonshine-ai/moonshine-streaming-medium)、[NVIDIA Nemotron 3.5 模型卡](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)。",
             "2. MP3 仅在准备阶段通过 macOS 系统 afconvert 转成 16 kHz、单声道 PCM16 WAV；不向推理库添加 FFmpeg 依赖。解码时间单列，不计入 ASR 耗时。",
             f"3. 完整音频按 {report['settings']['segment_seconds']} 秒、无重叠窗口顺序请求 `/v1/transcribe`；最后不足一窗的样本全部保留。样本覆盖校验保存在 scored-results.json 的 coverage 字段。该测试为长文件分段离线吞吐，不是实时采集的端到端字幕延迟。固定边界可能切断单词/句子，错误归于本套推理流程。",
             f"4. threads={report['settings']['threads']}、max_tokens={report['settings']['max_tokens']}；每个模型/backend/文件独立启动进程，先预热首窗 {report['settings']['warmup']} 次，再完整推理 {report['settings']['iterations']} 次。计时使用 perf_counter，包含 HTTP、模型计算和必要循环，不含预热/加载/解码。RTF=推理秒数÷音频秒数，越低越好。",
             "5. 峰值内存为该独立 server 进程退出时 Darwin wait4 返回的 ru_maxrss（字节转 MiB），涵盖加载、预热和正式推理；不是 Python 父进程累计内存，也不是单次窗口的内存增量。完成后主动 SIGTERM 回收子进程，因此记录中的 exit_code=-15 是计划内关闭，不是识别崩溃。Metal 使用统一内存，RSS 不等于独立 GPU 显存或完整系统物理 footprint。",
             "6. 初次运行在文件遇到错误后停止该文件；仅对失败配置进行补测，补测继续尝试全部窗口。原始失败未删除，合并行的 first_attempt 保留首次信息。表格使用全覆盖补测的耗时/内存，失败窗口的请求时间仍计入，不用失败的 0 秒占位作为性能。预热失败也会记录。\n7. 加载耗时含设备初始化与健康检查。测试前计算模型 SHA-256，会预热文件缓存，因此不得称为磁盘冷启动。每个文件仅正式全量一轮时，不提供置信区间或性能 p95；其他桌面进程未被强制关闭。执行顺序为先 Metal 后 CPU，模型顺序固定，未做随机化；单轮结果不用于判定微小性能差异。", "",
             "## WER / CER 预估的参考依据", "",
             "**没有人工校对真值时，无法得到真实 WER/CER。下表带星号的 `WER*` / `CER*` 是伪参考差异率，用于粗略预估和排查，不能视为准确率、排行榜或置信区间。多个模型可能共同漏字或共同幻觉，低差异并不证明正确；数值也可能超过 100%。**", "",
             "- 优先读取显式指定参考目录内同路径 `.txt`；没有对应文稿时，按 20 秒窗口从其他模型中选择字符归一化距离最小的 medoid 作为伪参考。始终排除被评估模型的全部 backend，避免拿自己当真值。CPU / Metal 共用同一份伪参考，优先选取其他模型的 Metal 输出。",
             "- 中文投票只允许 X-ASR、Confucius 和 Nemotron，英文 Moonshine 不参与中文伪真值选择。中文候选少于英文，留一后可能仅有两款模型，平分按稳定的文件名字典序选取；不能凭此分数认定哪款模型更准确。Moonshine 的中文分数仅展示本次越界输入的表现。",
             "- 文本归一化：Unicode NFKC、忽略大小写和标点；不做数字读法、简繁转换或人工语义修正。CER 按字母/数字/汉字计字符，排除空格；英文 WER 按词计数，中文 WER 为“单汉字 + 连续英文/数字词”的混合 token 预估，不是经过中文词法分词的标准 WER。",
             "- 公式：编辑距离÷参考单位数；汇总按参考单位数加权。空参考的插入仍计入总编辑数，参考总长为零时显示 —。每个窗口的参考原文、来源模型和原始输出均已保存，可替换人工真值后重新评分。补测中的 HTTP 失败窗口以空输出计入整条推理流程的漏识别，同时列明失败窗数量；这不等同于模型在解除生成限制后的识别能力。", "",
             "## 模型汇总（中英文分列）", "",
             "| Backend | 模型 | 语种 | 文件数 | 音频 s | 推理 s | RTF | 峰值 RSS MiB | WER* | CER* | 失败窗 |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for backend in report["settings"]["backend"]:
        for model in report["models"]:
            name = Path(model["path"]).name
            for language in ["en", "zh"]:
                value = summary([r for r in report["results"] if r["backend"] == backend and r["model"] == name and r["audio"].startswith(language + "/")])
                if value["files"]:
                    lines.append(f"| {backend} | {name} | {language} | {value['files']} | {value['audio_s']:.3f} | {value['inference_s']:.3f} | {value['rtf']:.4f} | {value['peak_rss_mib']:.1f} | {percent(value['wer'])} | {percent(value['cer'])} | {value['failed_segments']} |")
    lines += ["", "## CPU / Metal 输出一致性", "", "下表仅比较两个 backend 的输出差异，不是相对音频真值的错误率；差异率以 Metal 归一化字符数为分母，包含失败窗口的空输出。", "",
              "| 模型 | 语种 | 文件数 | 全文逐字相同文件数 | CPU↔Metal 字符差异率 |", "|---|---|---:|---:|---:|"]
    by_key={(r["backend"],r["model"],r["audio"]):r for r in report["results"] if r["status"] in COMPLETED}
    for model in report["models"]:
        name=Path(model["path"]).name
        for language in ["en","zh"]:
            pairs=[(by_key[("metal",name,a["relative"])],by_key[("cpu",name,a["relative"])]) for a in report["audio"]
                   if a["relative"].startswith(language+"/") and ("metal",name,a["relative"]) in by_key and ("cpu",name,a["relative"]) in by_key]
            if pairs:
                length=sum(len(tokens(left["text"],"cer")) for left,right in pairs)
                edits=sum(distance(tokens(left["text"],"cer"),tokens(right["text"],"cer")) for left,right in pairs)
                exact=sum(left["text"]==right["text"] for left,right in pairs)
                lines.append(f"| {name} | {language} | {len(pairs)} | {exact} | {percent(edits/length if length else None)} |")
    lines += ["", "## 逐文件结果", "", "加载/预热与正式推理分开；进程总耗时包含加载、预热、全部正式轮次和关闭。", "",
              "| Backend | 模型 | 文件 | 音频 s | 加载 s | 预热 s | 推理 s | 进程总 s | RTF | RSS MiB | WER* | CER* | 空窗 | 失败窗 |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["results"]:
        if row["status"] not in COMPLETED:
            lines.append(f"| {row['backend']} | {row['model']} | {row['audio']} | ERROR: {row.get('error')} |")
            continue
        transcript = output / row["artifact_dir"] / "transcript.txt"
        q = row["quality"]
        lines.append(f"| {row['backend']} | {row['model']} | [{row['audio']}]({relative(transcript)}) | {row['audio_s']:.3f} | {row['startup_s']:.3f} | {sum(row['warmup_s']):.3f} | {row['inference_mean_s']:.3f} | {row['process_elapsed_s']:.3f} | {row['rtf']:.4f} | {row['peak_rss_bytes']/2**20:.1f} | {percent(q['wer'])} | {percent(q['cer'])} | {row['empty_segments']} | {row.get('failed_segments',0)} |")
    retried=[r for r in report["results"] if "first_attempt" in r]
    if retried:
        lines += ["", "## 首次失败与全覆盖补测", "", "原始失败记录原样保留，未解除 512 token 限制；补测仅改变遇错后的继续执行策略。", "",
                  "| Backend | 模型 | 文件 | 首次错误 | 补测失败窗 | 原始日志 |", "|---|---|---|---|---:|---|"]
        for row in retried:
            first=row["first_attempt"]
            log=Path(first["artifact_dir"]) / "server.log"
            lines.append(f"| {row['backend']} | {row['model']} | {row['audio']} | {first.get('error','')} | {row.get('failed_segments',0)} | [log]({relative(log)}) |")
    lines += ["", "## 输入清单与解码", "", "| 音频 | 完整时长 s | PCM 样本数 | afconvert s |", "|---|---:|---:|---:|"]
    for audio in report["audio"]:
        lines.append(f"| {audio['relative']} | {audio['audio_s']:.6f} | {audio['samples']} | {audio['conversion_s']:.3f} |")
    lines += ["", "## 原始证据与复现", "",
              f"- [完整测量 JSON]({relative(output / 'results.json')})、[含逐窗参考的评分 JSON]({relative(output / 'scored-results.json')})、[CSV]({relative(output / 'summary.csv')})。",
              "- 每个测试目录保留 `transcript.txt`、`segments.jsonl`、`metrics.json`、`server.log`；JSON 包含全部样本覆盖、逐窗时长、退出状态与完整文本。",
              f"- server SHA-256：`{report['server']['sha256']}`。", ""]
    for model in report["models"]:
        lines.append(f"- `{Path(model['path']).name}`：{model['bytes']} bytes，SHA-256 `{model['sha256']}`。")
    lines += ["", "执行参数（路径以原始 JSON 为准）：", "", "```json", json.dumps(report["settings"], ensure_ascii=False, indent=2), "```", ""]
    notes=["## 本次性能观察", "", "以下仅总结本机本次测量。质量列是模型互参预估，不是人工真值 WER/CER。", ""]
    expected_en=sum(a["relative"].startswith("en/") for a in report["audio"])
    for model in report["models"]:
        name=Path(model["path"]).name
        groups={backend:summary([r for r in report["results"] if r["model"]==name and r["backend"]==backend and r["audio"].startswith("en/")]) for backend in ["cpu","metal"]}
        cpu,metal=groups["cpu"],groups["metal"]
        if expected_en and cpu["files"]==expected_en and metal["files"]==expected_en:
            winner="CPU" if cpu["inference_s"]<metal["inference_s"] else "Metal"
            factor=max(cpu["inference_s"],metal["inference_s"])/min(cpu["inference_s"],metal["inference_s"])
            notes.append(f"- `{name}`：{expected_en} 个英文文件 CPU 共 {cpu['inference_s']:.3f}s（RTF {cpu['rtf']:.4f}），Metal 共 {metal['inference_s']:.3f}s（RTF {metal['rtf']:.4f}）；本次 {winner} 吞吐约为另一后端的 {factor:.2f} 倍。")
    notes += ["- 中文场景请查看独立的 zh 行；不把英文模型的中文空输出或失败当成高性能。参考文本中的数字写法和专有名词尚未人工归一化，不能按预估分数认定实际识别质量高低。", ""]
    lines[lines.index("## 测试方法与指标边界"):lines.index("## 测试方法与指标边界")]=notes
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--retry-results", type=Path, action="append", default=[])
    parser.add_argument("--artifacts-dir", type=Path)
    args = parser.parse_args()
    report=json.loads(args.results.read_text())
    output=args.artifacts_dir or args.results.parent
    output.mkdir(parents=True,exist_ok=True)
    if args.retry_results:
        if output.resolve() == args.results.parent.resolve(): raise ValueError("use a separate --artifacts-dir to preserve the baseline")
        report=merge_results(report,args.results.parent,[(json.loads(path.read_text()),path.parent) for path in args.retry_results])
        report["source_runs"]=[str(args.results.resolve())]+[str(path.resolve()) for path in args.retry_results]
        (output / "results.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    render(report, args.report, output, args.reference_dir)


if __name__ == "__main__":
    main()
