#!/usr/bin/env python3
"""Build, test, audit and package one native release target."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
import struct
import subprocess
import sys

from release_common import CUDA_DLLS, package, read_version

ROOT = Path(__file__).resolve().parents[1]
SYSTEM_DLLS = set("kernel32 user32 advapi32 bcrypt crypt32 ntdll ws2_32 wsock32 iphlpapi secur32 ole32 oleaut32 shell32 shlwapi winmm version normaliz rpcrt4 dbghelp psapi setupapi mswsock ucrtbase msvcrt comdlg32 gdi32".split())


def build_plan(root, directory, system, arch, jobs, variant="cpu"):
    if variant not in {"cpu", "vulkan", "cuda126"} or (system != "windows" and variant != "cpu"):
        raise ValueError("unsupported backend variant for platform")
    if jobs < 1:
        raise ValueError("jobs must be positive")
    if (system, arch) not in {("macos", "arm64"), ("macos", "x86_64"), ("windows", "x64")}:
        raise ValueError("unsupported platform/architecture")
    configure = ["cmake", "-S", str(root), "-B", str(directory),
                 "-DCMAKE_BUILD_TYPE=Release", "-DHAOLIVE_BUILD_TESTS=ON",
                 "-DHAOLIVE_BUILD_SHARED=OFF", "-DHAOLIVE_VULKAN=" + ("ON" if variant == "vulkan" else "OFF"),
                 "-DHAOLIVE_CUDA=" + ("ON" if variant == "cuda126" else "OFF"),
                 "-DHAOLIVE_METAL=" + ("ON" if system == "macos" else "OFF"),
                 "-DPython3_EXECUTABLE=" + sys.executable]
    if system == "macos":
        configure += ["-DCMAKE_OSX_ARCHITECTURES=" + arch, "-DCMAKE_OSX_DEPLOYMENT_TARGET=13.0"]
    elif variant == "cuda126":
        configure += ["-G", "Ninja", "-DCMAKE_C_COMPILER=cl", "-DCMAKE_CXX_COMPILER=cl"]
    else:
        configure += ["-G", "Visual Studio 17 2022", "-A", "x64"]
    if variant == "cuda126":
        configure += ["-DHAOLIVE_CUDA_126=ON", "-DGGML_STATIC=ON", "-DGGML_CUDA_NO_VMM=ON",
                      "-DCMAKE_CUDA_ARCHITECTURES=75-real;80-virtual;86-real;89-real"]
    if arch in {"x64", "x86_64"}:
        configure += ["-DGGML_SSE42=ON", "-DGGML_AVX=ON", "-DGGML_AVX2=ON", "-DGGML_FMA=ON", "-DGGML_F16C=ON", "-DGGML_BMI2=ON"]
    return [configure,
            ["cmake", "--build", str(directory), "--config", "Release", "--parallel", str(jobs)],
            ["ctest", "--test-dir", str(directory), "-C", "Release", "--output-on-failure", "--no-tests=error"]]


def pe_imports(data):
    """Read PE32+ imports without requiring dumpbin to be on PATH."""
    try:
        return _pe_imports(data)
    except (struct.error, IndexError, UnicodeDecodeError) as error:
        raise ValueError("invalid PE import table") from error


def _pe_imports(data):
    if data[:2] != b"MZ":
        raise ValueError("not a Windows executable")
    header = struct.unpack_from("<I", data, 0x3c)[0]
    if data[header:header + 4] != b"PE\0\0" or struct.unpack_from("<H", data, header + 4)[0] != 0x8664:
        raise ValueError("expected Windows x64 executable")
    count = struct.unpack_from("<H", data, header + 6)[0]
    size = struct.unpack_from("<H", data, header + 20)[0]
    optional = header + 24
    if struct.unpack_from("<H", data, optional)[0] != 0x20b or size < 240:
        raise ValueError("expected PE32+ optional header")
    sections = optional + size

    def offset(rva):
        for index in range(count):
            entry = sections + index * 40
            virtual_size, start, raw_size, raw = struct.unpack_from("<IIII", data, entry + 8)
            if start <= rva < start + max(virtual_size, raw_size) and rva - start < raw_size:
                result = raw + rva - start
                if result < len(data):
                    return result
        raise ValueError("invalid PE RVA")

    delayed = struct.unpack_from("<I", data, optional + 112 + 13 * 8)[0]
    if delayed:
        raise ValueError("unexpected delayed DLL imports")
    imports = struct.unpack_from("<I", data, optional + 120)[0]
    if not imports:
        return []
    result = []
    start = offset(imports)
    for index in range(256):
        descriptor = struct.unpack_from("<IIIII", data, start + index * 20)
        if not any(descriptor):
            return result
        name_at = offset(descriptor[3])
        end = data.index(b"\0", name_at, min(len(data), name_at + 256))
        result.append(data[name_at:end].decode("ascii").lower())
    raise ValueError("unbounded PE import table")


def audit(binary, system, arch, number, variant="cpu", runtime_files=None, probe_devices=True):
    actual = subprocess.check_output([str(binary), "--version"], text=True, timeout=30).strip()
    if actual != "haolive-server " + number:
        raise ValueError("binary version does not match VERSION: " + actual)
    backends = json.loads(subprocess.check_output([str(binary), "--list-backends"], text=True, timeout=120)) if probe_devices else ["cpu"]
    expected = {"cpu", "metal"} if system == "macos" else {"cpu"} | ({"vulkan"} if variant == "vulkan" else {"cuda"} if variant == "cuda126" else set())
    compiled = json.loads(subprocess.check_output([str(binary), "--list-compiled-backends"], text=True, timeout=30))
    if set(compiled) != expected:
        raise ValueError("compiled backends do not match package variant: " + repr(compiled))
    if "cpu" not in backends or set(backends) - expected:
        raise ValueError("unexpected backends: " + repr(backends))
    if system == "macos":
        built_arch = subprocess.check_output(["lipo", "-archs", str(binary)], text=True).strip()
        if built_arch != arch:
            raise ValueError("archive architecture mismatch: " + built_arch)
        dependencies = subprocess.check_output(["otool", "-L", str(binary)], text=True).splitlines()[1:]
        if any(not line.strip().startswith(("/usr/lib/", "/System/Library/")) for line in dependencies):
            raise ValueError("server has non-system dynamic dependencies")
    else:
        files = {"haolive-server.exe": binary, **(runtime_files or {})}
        packaged = {name.lower() for name in files}
        driver = {"vulkan-1.dll"} if variant == "vulkan" else set()
        for source in files.values():
            if source.suffix.lower() not in {".dll", ".exe"}:
                continue
            for name in pe_imports(source.read_bytes()):
                stem = name.removesuffix(".dll")
                if not name.endswith(".dll") or not (name in packaged or name in driver or stem in SYSTEM_DLLS or stem.startswith(("api-ms-win-", "ext-ms-win-"))):
                    raise ValueError("non-system DLL dependency: " + name)


def cuda_runtime_files(root):
    root = Path(root)
    metadata = json.loads((root / "version.json").read_text(encoding="utf-8"))
    if not metadata["cuda"]["version"].startswith("12.6."):
        raise ValueError("cuda126 requires CUDA Toolkit 12.6.x")
    files = {}
    for name in CUDA_DLLS:
        matches = [p for p in (root / "bin").glob("*.dll") if p.name.lower() == name.lower()]
        if matches:
            files[name] = matches[0]
    for required in ("cublas64_12.dll", "cublasLt64_12.dll"):
        if required not in files:
            raise ValueError("CUDA runtime missing " + required)
    candidates = [root / "EULA.txt", root / "doc" / "EULA.txt"]
    license_path = next((p for p in candidates if p.is_file()), None)
    if not license_path:
        raise ValueError("CUDA Toolkit EULA.txt is required for redistribution")
    files["CUDA-LICENSE.txt"] = license_path
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True, choices=["macos", "windows"])
    parser.add_argument("--arch")
    parser.add_argument("--variant", choices=["cpu", "vulkan", "cuda126"], default="cpu")
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--jobs", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-device-probe", action="store_true", help="Skip GPU enumeration on virtual CI hosts; compiled backends and CTest remain required")
    args = parser.parse_args(argv)
    arch = args.arch or ("arm64" if platform.machine().lower() in {"arm64", "aarch64"} else "x86_64" if args.platform == "macos" else "x64")
    number = read_version(ROOT)
    target = f"{args.platform}-{arch}" + ("-" + args.variant if args.platform == "windows" else "")
    directory = (args.build_dir or ROOT / "build-release" / target).resolve()
    plan = build_plan(ROOT, directory, args.platform, arch, args.jobs, args.variant)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    expected_host = "darwin" if args.platform == "macos" else "win32"
    if sys.platform != expected_host:
        raise ValueError("run this platform build on its native operating system")
    for command in plan:
        subprocess.run(command, cwd=ROOT, check=True)
    binary = directory / "bin" / "haolive-server"
    if args.platform == "windows":
        binary = directory / ("bin/haolive-server.exe" if args.variant == "cuda126" else "bin/Release/haolive-server.exe")
    extras = {}
    if args.variant == "cuda126":
        cuda_root = os.environ.get("CUDA_PATH", "")
        if not cuda_root:
            raise ValueError("CUDA_PATH must point to CUDA Toolkit 12.6")
        extras = cuda_runtime_files(cuda_root)
    # Verify a distribution-shaped copy with SDK directories removed from PATH.
    with tempfile.TemporaryDirectory(prefix="haolive-package-") as temp:
        staged = Path(temp) / binary.name
        shutil.copy2(binary, staged)
        for name, source in extras.items():
            shutil.copy2(source, Path(temp) / name)
        if args.platform == "windows":
            original_path = os.environ.get("PATH", "")
            windows = Path(os.environ["SystemRoot"])
            os.environ["PATH"] = os.pathsep.join([str(windows / "System32"), str(windows)])
        try:
            audit(staged, args.platform, arch, number, args.variant,
                  {name: Path(temp) / name for name in extras}, not args.skip_device_probe)
        finally:
            if args.platform == "windows":
                os.environ["PATH"] = original_path
    archive = package(binary, args.output_dir.resolve(), number, target, extras)
    print("Release asset: " + str(archive))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        sys.exit("Build failed: " + str(error))
