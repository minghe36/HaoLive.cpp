#!/usr/bin/env python3
"""Build, test, audit and package one native release target."""
import argparse
import json
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys

from release_common import package, read_version

ROOT = Path(__file__).resolve().parents[1]
SYSTEM_DLLS = set("kernel32 user32 advapi32 bcrypt crypt32 ntdll ws2_32 iphlpapi secur32 ole32 oleaut32 shell32 shlwapi winmm version normaliz rpcrt4 dbghelp psapi setupapi mswsock ucrtbase msvcrt comdlg32 gdi32".split())


def build_plan(root, directory, system, arch, jobs):
    if jobs < 1:
        raise ValueError("jobs must be positive")
    if (system, arch) not in {("macos", "arm64"), ("macos", "x86_64"), ("windows", "x64")}:
        raise ValueError("unsupported platform/architecture")
    configure = ["cmake", "-S", str(root), "-B", str(directory),
                 "-DCMAKE_BUILD_TYPE=Release", "-DHAOLIVE_BUILD_TESTS=ON",
                 "-DHAOLIVE_BUILD_SHARED=OFF", "-DHAOLIVE_VULKAN=OFF", "-DHAOLIVE_CUDA=OFF",
                 "-DHAOLIVE_METAL=" + ("ON" if system == "macos" else "OFF"),
                 "-DPython3_EXECUTABLE=" + sys.executable]
    if system == "macos":
        configure += ["-DCMAKE_OSX_ARCHITECTURES=" + arch, "-DCMAKE_OSX_DEPLOYMENT_TARGET=13.0"]
    else:
        configure += ["-G", "Visual Studio 17 2022", "-A", "x64"]
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


def audit(binary, system, arch, number):
    actual = subprocess.check_output([str(binary), "--version"], text=True, timeout=30).strip()
    if actual != "haolive-server " + number:
        raise ValueError("binary version does not match VERSION: " + actual)
    backends = json.loads(subprocess.check_output([str(binary), "--list-backends"], text=True, timeout=120))
    expected = {"cpu", "metal"} if system == "macos" else {"cpu"}
    if "cpu" not in backends or set(backends) - expected or (system == "windows" and set(backends) != expected):
        raise ValueError("unexpected backends: " + repr(backends))
    if system == "macos":
        built_arch = subprocess.check_output(["lipo", "-archs", str(binary)], text=True).strip()
        if built_arch != arch:
            raise ValueError("archive architecture mismatch: " + built_arch)
        dependencies = subprocess.check_output(["otool", "-L", str(binary)], text=True).splitlines()[1:]
        if any(not line.strip().startswith(("/usr/lib/", "/System/Library/")) for line in dependencies):
            raise ValueError("server has non-system dynamic dependencies")
    else:
        for name in pe_imports(binary.read_bytes()):
            stem = name.removesuffix(".dll")
            if not name.endswith(".dll") or not (stem in SYSTEM_DLLS or stem.startswith(("api-ms-win-", "ext-ms-win-"))):
                raise ValueError("non-system DLL dependency: " + name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True, choices=["macos", "windows"])
    parser.add_argument("--arch")
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--jobs", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    arch = args.arch or ("arm64" if platform.machine().lower() in {"arm64", "aarch64"} else "x86_64" if args.platform == "macos" else "x64")
    number = read_version(ROOT)
    target = f"{args.platform}-{arch}"
    directory = (args.build_dir or ROOT / "build-release" / target).resolve()
    plan = build_plan(ROOT, directory, args.platform, arch, args.jobs)
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
        binary = directory / "bin/Release/haolive-server.exe"
    audit(binary, args.platform, arch, number)
    archive = package(binary, args.output_dir.resolve(), number, target)
    print("Release asset: " + str(archive))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        sys.exit("Build failed: " + str(error))
