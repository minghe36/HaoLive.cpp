# 构建与发布

## 版本与产物

`VERSION` 是版本号的唯一来源，格式为 `MAJOR.MINOR.PATCH`，例如 `0.1.0`。
CMake 和二进制 `haolive-server --version` 使用同一版本。更新日志位于
`changelog/<版本号>.md`，Git tag 为 `v<版本号>`，Release 正文直接读取该文件。

| 平台                  | 构建环境                         | 发布包                                   | 后端         |
| --------------------- | -------------------------------- | ---------------------------------------- | ------------ |
| macOS arm64           | macOS / Xcode Command Line Tools | `HaoLive-<版本>-macos-arm64.tar.gz`      | CPU + Metal  |
| macOS Intel           | macOS / Xcode Command Line Tools | `HaoLive-<版本>-macos-x86_64.tar.gz`     | CPU + Metal  |
| Windows x64 CPU       | VS 2022 + 对应 SDK               | `HaoLive-<版本>-windows-x64-cpu.zip`     | CPU          |
| Windows x64 Vulkan    | VS 2022 + 对应 SDK               | `HaoLive-<版本>-windows-x64-vulkan.zip`  | CPU + Vulkan |
| Windows x64 CUDA 12.6 | VS 2022 + 对应 SDK               | `HaoLive-<版本>-windows-x64-cuda126.zip` | CPU + CUDA   |

CPU、macOS 和 Vulkan 包只含可执行文件；CUDA 包另含必需的 CUDA/cuBLAS DLL 与 `CUDA-LICENSE.txt`。每个包另上传同名 `.sha256` 校验文件。
权重不打包。第三方许可证已嵌入程序，通过 `--licenses` 查看。macOS 最低部署目标为
13.0；x64 构建启用 SSE4.2、AVX2、FMA、F16C 和 BMI2，需要相应 CPU 指令集。
Windows 使用 UTF-8 源码/执行字符集并静态链接 MSVC 运行库，禁止 OpenMP 和动态 ggml 后端。CUDA 版仅允许指定的 NVIDIA 运行库，递归检查打包 DLL 的导入表。Vulkan loader 来自 GPU 驱动。

## 本地构建

需要 CMake 3.24+、Python 3.10+ 和对应平台编译器。脚本只使用 Python 标准库。
从任意目录调用脚本均可，路径可以包含空格。

```sh
# macOS 默认使用本机架构
./scripts/build-macos.sh
./scripts/build-macos.sh --arch arm64 --jobs 4
./scripts/build-macos.sh --arch x86_64 --output-dir ./dist-intel

# 仅查看 CMake/CTest 命令
./scripts/build-macos.sh --dry-run
```

```powershell
# Windows PowerShell，需要 python 在 PATH 中
./scripts/build-windows.ps1
./scripts/build-windows.ps1 -Architecture x64 -Variant cpu
./scripts/build-windows.ps1 -Architecture x64 -Variant vulkan -Jobs 2
./scripts/build-windows.ps1 -Architecture x64 -Variant cuda126 -Jobs 2
```

Windows GPU 构建需要对应 SDK；CUDA 固定为 12.6.x 并使用 v142 工具集。使用 `VULKAN_SDK` / `CUDA_PATH` 指定实际 SDK 路径。

默认构建目录为仓库下 `build-release/<平台-架构[-后端]>/`，产物写入 `dist/`。
脚本依次执行 Release 配置、编译、CTest、版本/架构/依赖审计、打包和 SHA-256 生成。
已有同名包时拒绝覆盖，重复构建请指定新的输出目录。macOS 交叉架构构建仍需要能够
运行目标架构的测试；CI 分别使用原生 arm64 和 Intel runner。

macOS 检查只允许系统动态库。Windows 使用 PE 导入表检查，拒绝独立 MSVC/OpenMP/
未打包的第三方运行时 DLL。无 Metal 设备的构建机可以完成 CPU 测试，但 Metal 编译通过不能
代替实际 GPU 推理验收。脚本不签名、不公证，不宣称已验证模型在所有平台的识别结果。

## 一条命令创建新版本

先审查、提交并推送需要发布的源码，包括脚本、工作流、第三方源码与 VERSION。
工作区必须干净，本地 HEAD 必须等于 origin 上当前分支的 HEAD。需要 Git、已登录的
GitHub CLI (`gh auth login`) 和 origin 仓库的推送/Release 权限。

把本次更新内容写入一个 UTF-8 文件（建议放在仓库外，避免被脏工作区检查阻止）。
例如文件内容为真实的变更条目；脚本会添加版本标题，不会自动编造更新日志。

```sh
# 0.1.1 仅为示例；替换成目标版本和实际更新日志文件
python3 scripts/release.py create 0.1.1 --notes-file /path/to/release-notes.md --dry-run
python3 scripts/release.py create 0.1.1 --notes-file /path/to/release-notes.md
```

这条命令将：

1. 检查版本递增、日志非空、工作区干净、远程分支同步、tag 尚不存在及 GitHub 可访问。
2. 写入 `VERSION` 和 `changelog/0.1.1.md`，只提交这两个文件。
3. 创建带更新日志的 annotated tag `v0.1.1`。
4. 用 `git push --atomic` 同时推送版本提交和 tag，不移动或强推已有 tag。
5. tag 触发 `.github/workflows/release.yml`，构建两个 macOS 和三个 Windows 后端产物。
6. 全部构建和测试成功后，创建 Draft Release，上传五个程序包与五个校验文件。
7. 核对远端资产的名称、大小和可用的 SHA-256 digest，再公开 Release。

命令返回的 Actions 链接表示已触发构建；只有工作流 Publish 步骤成功才代表 Release
已发布。GitHub Token 通过工作流环境提供，脚本不会将 Token 写入源码或命令参数。
GitHub Actions 需要启用，且仓库/组织策略必须允许该工作流写入 Releases。

Windows 将发布命令中的 `python3` 替换为 `python`，并使用实际 Windows 更新日志路径。
不要把本机只构建的一个平台包作为完整跨平台 Release。

## 分步审查与失败恢复

需要先审查版本文件时：

```sh
python3 scripts/release.py prepare 0.1.1 --notes-file /path/to/release-notes.md
# 审查后只提交这些版本文件，并推送到 origin
# git add VERSION changelog/0.1.1.md
# git commit -m 'Release v0.1.1'
# git push origin HEAD
python3 scripts/release.py publish --dry-run
python3 scripts/release.py publish
```

受保护分支可使用 `prepare`，经 PR 合并后在目标提交执行 `publish`。
`prepare` 不创建提交或 tag。`publish` 使用当前已提交、已推送的 VERSION；已有本地
annotated tag 时只接受它指向当前 HEAD，不覆盖错误 tag。

- 本地提交/tag 创建后推送失败：保留本地结果供检查。修复原因后推送已审查的分支，
  再执行 `publish`。脚本不会自动 reset、删除提交、移动 tag 或清理用户文件。
- CI 构建失败：不会创建 Release。修复源码应创建新版本；临时 runner 故障可重跑原工作流。
- 上传中断：Release 保持 Draft；在 Actions 重跑 Publish 失败任务，或手动运行 Release
  工作流并输入原 tag。只允许重传 Draft 中的预期资产，公开 Release 不会被覆盖。
- 校验失败、缺包、版本/tag 不一致：拒绝公开 Release，检查工作流日志后处理。

发布模式的 CI 重试输入必须是现有 annotated tag，版本文件和更新日志必须属于该 tag 的提交。
构建阶段只有 contents:read，只有 Publish 阶段具有 contents:write；第三方 Actions
固定到完整 commit SHA。

## 发布前 CI 验证

手动运行 Release，设置 `tag=master`、`build_only=true`，可从分支构建并检查五种包，不创建 Release。验证通过后再创建正式版本 tag。发布模式默认 `build_only=false`，仍要求 annotated tag、版本文件、日志与提交一致。

`--list-compiled-backends` 核对已编译后端，`--list-backends` 核对本机可用设备。无 GPU 的 runner 不要求枚举出 GPU；编译和 CPU 测试不能代替真实 GPU 模型验收。

## 验证脚本

```sh
python3 -m unittest discover -s tests -p test_release.py -v
ctest --test-dir build-release/macos-arm64 --output-on-failure
```

脚本测试使用临时 Git 仓库和本地 bare remote，验证真实的版本提交、annotated tag 与
原子推送；GitHub Release 操作用模拟响应验证完整上传、Draft 重试和失败阻断，测试
不会向真实 GitHub 仓库提交、打 tag 或发布。

CI runner 标签依据 [GitHub 官方 runner 列表](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)。
Draft、tag 验证与资产上传使用 [gh release create](https://cli.github.com/manual/gh_release_create)
和 [gh release upload](https://cli.github.com/manual/gh_release_upload)。

## 本次本机验证（2026-10-20）

- macOS arm64：新构建脚本从 Release 配置到打包全部通过。
- macOS x86_64：跨架构编译通过，测试及解包程序在 Rosetta 下通过；未在 Intel 实机验收。
- 两个目标的五组 CTest 全部通过，其中发布脚本测试覆盖临时 Git 远程的真实原子推送，
  以及 GitHub 上传失败、Draft 重试、校验失败、公开版本保护。
- 两个平台的包已验证 SHA-256、单文件结构、架构、`--version` 和许可证输出。
- 工作流通过 actionlint v1.7.12；macOS shell 入口通过 `bash -n`。
- Windows 构建命令、PE 依赖审计与失败路径已测试；当前没有 Windows 执行环境，
  Windows 编译/运行及真实 GitHub Release 发布仍需工作流执行后确认。
- 当前工作区包含未提交/未跟踪的源码，实际版本准备 dry-run 已验证会阻止修改。
  本次没有在真实仓库创建提交、tag、推送或 Release。
