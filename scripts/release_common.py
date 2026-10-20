"""Shared version and archive contract for local and CI releases (stdlib only)."""
import hashlib
from pathlib import Path
import re
import tarfile
import zipfile

TARGETS = ("macos-arm64", "macos-x86_64", "windows-x64")


def version(value):
    value = value.removeprefix("v")
    if not re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", value):
        raise ValueError("version must be MAJOR.MINOR.PATCH, optionally prefixed with v")
    return value


def read_version(root):
    raw = (Path(root) / "VERSION").read_text(encoding="utf-8").strip()
    number = version(raw)
    if raw != number:
        raise ValueError("VERSION must omit the v prefix")
    return number


def asset_name(number, target):
    if target not in TARGETS:
        raise ValueError("unsupported release target: " + target)
    extension = ".zip" if target.startswith("windows") else ".tar.gz"
    return f"HaoLive-{version(number)}-{target}{extension}"


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package(binary, output, number, target):
    binary, output = Path(binary), Path(output)
    if binary.is_symlink() or not binary.is_file() or binary.stat().st_size == 0:
        raise ValueError("missing or empty server binary")
    output.mkdir(parents=True, exist_ok=True)
    archive = output / asset_name(number, target)
    digest_file = archive.with_name(archive.name + ".sha256")
    if archive.exists() or digest_file.exists():
        raise FileExistsError("release asset already exists: " + str(archive))
    name = "haolive-server.exe" if target.startswith("windows") else "haolive-server"
    if target.startswith("windows"):
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
            bundle.write(binary, name)
    else:
        with tarfile.open(archive, "x:gz") as bundle:
            info = bundle.gettarinfo(str(binary), arcname=name)
            info.mode = 0o755
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with binary.open("rb") as source:
                bundle.addfile(info, source)
    digest_file.write_text(f"{checksum(archive)}  {archive.name}\n", encoding="utf-8", newline="\n")
    return archive


def release_assets(directory, number):
    directory = Path(directory)
    expected = {asset_name(number, target) for target in TARGETS}
    expected |= {name + ".sha256" for name in expected.copy()}
    files = {p.name: p for p in directory.iterdir() if p.is_file()}
    if set(files) != expected:
        raise ValueError("release requires exactly all three platform archives and their checksums")
    for target in TARGETS:
        name = asset_name(number, target)
        archive = files[name]
        if archive.is_symlink() or files[name + ".sha256"].is_symlink():
            raise ValueError("release assets must not be symlinks")
        if files[name + ".sha256"].read_text(encoding="utf-8").strip() != f"{checksum(archive)}  {name}":
            raise ValueError("checksum mismatch: " + name)
        binary = "haolive-server.exe" if target.startswith("windows") else "haolive-server"
        if target.startswith("windows"):
            with zipfile.ZipFile(archive) as bundle:
                entries = bundle.infolist()
                valid = len(entries) == 1 and entries[0].filename == binary and entries[0].file_size > 0
        else:
            with tarfile.open(archive) as bundle:
                entries = bundle.getmembers()
                valid = len(entries) == 1 and entries[0].name == binary and entries[0].isfile() and entries[0].size > 0 and bool(entries[0].mode & 0o111)
        if not valid:
            raise ValueError("archive must contain only its executable: " + name)
    return [files[name] for name in sorted(files)]
