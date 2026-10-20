import contextlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build
import release
from release_common import TARGETS, asset_name, checksum, package, read_version, release_assets, version


class AssetTests(unittest.TestCase):
    def test_versions_and_targets_reject_invalid_or_ambiguous_names(self):
        self.assertEqual(version("v1.2.3"), "1.2.3")
        for invalid in ["01.2.3", "1.2", "1.2.3-beta", "../1.2.3", "1.2.3;touch x", "1.2.3\n", "v1.2.3x"]:
            with self.assertRaises(ValueError):
                version(invalid)
        with self.assertRaises(ValueError):
            asset_name("1.2.3", "windows-arm64")

    def make_assets(self, root):
        binary = root / "binary"
        binary.write_bytes(b"executable-fixture")
        for target in TARGETS:
            package(binary, root / "assets", "1.2.3", target)
        return root / "assets"

    def test_archives_preserve_only_executable_and_checksums(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            assets = self.make_assets(root)
            self.assertEqual(len(release_assets(assets, "1.2.3")), 6)
            for target in TARGETS:
                path = assets / asset_name("1.2.3", target)
                if target.startswith("windows"):
                    with zipfile.ZipFile(path) as bundle:
                        self.assertEqual(bundle.namelist(), ["haolive-server.exe"])
                        self.assertEqual(bundle.read("haolive-server.exe"), b"executable-fixture")
                else:
                    with tarfile.open(path) as bundle:
                        self.assertEqual(bundle.getnames(), ["haolive-server"])
                        self.assertEqual(bundle.getmembers()[0].mode, 0o755)
            with self.assertRaises(FileExistsError):
                package(root / "binary", assets, "1.2.3", TARGETS[0])
            first = assets / asset_name("1.2.3", TARGETS[0])
            first.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                release_assets(assets, "1.2.3")

    def test_missing_extra_and_unexpected_archive_contents_are_blocked(self):
        with tempfile.TemporaryDirectory() as temp:
            assets = self.make_assets(Path(temp))
            extra = assets / "unknown.zip"
            extra.touch()
            with self.assertRaises(ValueError):
                release_assets(assets, "1.2.3")
            extra.unlink()
            win = assets / asset_name("1.2.3", "windows-x64")
            with zipfile.ZipFile(win, "a") as bundle:
                bundle.writestr("models/weight.gguf", b"not-a-release-binary")
            win.with_name(win.name + ".sha256").write_text(f"{checksum(win)}  {win.name}\n")
            with self.assertRaisesRegex(ValueError, "only its executable"):
                release_assets(assets, "1.2.3")
            win.unlink()
            with self.assertRaises(ValueError):
                release_assets(assets, "1.2.3")


class BuildTests(unittest.TestCase):
    def test_platform_plans_enable_tests_and_portable_backends(self):
        mac = build.build_plan(Path("source with spaces"), Path("build with spaces"), "macos", "arm64", 3)
        self.assertIn("-DHAOLIVE_METAL=ON", mac[0])
        self.assertIn("-DCMAKE_OSX_DEPLOYMENT_TARGET=13.0", mac[0])
        self.assertIn("--no-tests=error", mac[2])
        win = build.build_plan(Path("source"), Path("build"), "windows", "x64", 4)
        self.assertIn("Visual Studio 17 2022", win[0])
        self.assertIn("-DHAOLIVE_METAL=OFF", win[0])
        self.assertFalse(any("OSX" in arg for arg in win[0]))
        with self.assertRaises(ValueError):
            build.build_plan(Path(), Path(), "windows", "arm64", 4)
        with self.assertRaises(ValueError):
            build.build_plan(Path(), Path(), "macos", "arm64", 0)

    def pe(self, name=b"KERNEL32.dll"):
        data = bytearray(1024)
        data[:2] = b"MZ"
        struct.pack_into("<I", data, 0x3c, 0x80)
        data[0x80:0x84] = b"PE\0\0"
        struct.pack_into("<HH", data, 0x84, 0x8664, 1)
        struct.pack_into("<H", data, 0x94, 240)
        struct.pack_into("<H", data, 0x98, 0x20b)
        struct.pack_into("<I", data, 0x98 + 120, 0x1000)
        struct.pack_into("<IIII", data, 0x98 + 240 + 8, 512, 0x1000, 512, 0x200)
        struct.pack_into("<I", data, 0x200 + 12, 0x1040)
        data[0x240:0x240 + len(name) + 1] = name + b"\0"
        return bytes(data)

    def test_pe_architecture_imports_and_dll_audit(self):
        self.assertEqual(build.pe_imports(self.pe()), ["kernel32.dll"])
        invalid = bytearray(self.pe())
        struct.pack_into("<H", invalid, 0x84, 0xaa64)
        with self.assertRaises(ValueError):
            build.pe_imports(invalid)
        with self.assertRaises(ValueError):
            build.pe_imports(b"MZ")
        with tempfile.TemporaryDirectory() as temp:
            binary = Path(temp) / "haolive-server.exe"
            binary.write_bytes(self.pe(b"WSOCK32.dll"))
            with patch.object(build.subprocess, "check_output", side_effect=["haolive-server 1.2.3", '["cpu"]']):
                build.audit(binary, "windows", "x64", "1.2.3")
            binary.write_bytes(self.pe(b"VCRUNTIME140.dll"))
            with patch.object(build.subprocess, "check_output", side_effect=["haolive-server 1.2.3", '["cpu"]']):
                with self.assertRaisesRegex(ValueError, "non-system DLL"):
                    build.audit(binary, "windows", "x64", "1.2.3")


class GitReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "source with spaces"
        self.root.mkdir()
        self.remote = Path(self.temp.name) / "remote.git"
        self.git("init", "-b", "main")
        self.git("config", "user.name", "Release Test")
        self.git("config", "user.email", "release@example.invalid")
        hooks = Path(self.temp.name) / "empty-hooks"
        hooks.mkdir()
        self.git("config", "core.hooksPath", str(hooks))
        self.git("config", "commit.gpgSign", "false")
        self.git("config", "tag.gpgSign", "false")
        (self.root / "VERSION").write_text("0.1.0\n")
        for filename in [".github/workflows/release.yml", "scripts/build.py", "scripts/release.py"]:
            path = self.root / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n")
        self.git("add", ".")
        self.git("commit", "-m", "Initial source")
        subprocess.run(["git", "init", "--bare", str(self.remote)], check=True, capture_output=True)
        self.git("remote", "add", "origin", str(self.remote))
        self.git("push", "-u", "origin", "main")
        self.notes = Path(self.temp.name) / "notes.md"
        self.notes.write_text('修复分块。 Literal `code` and $(not-a-command).\n', encoding="utf-8")
        self.original_output = release.output

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.root, text=True, encoding="utf-8", stderr=subprocess.PIPE).strip()

    def output(self, root, *args):
        if args[:4] == ("git", "remote", "get-url", "origin"):
            return "git@github.com:example/HaoLive.cpp.git"
        if args[0] == "gh":
            return "example/HaoLive.cpp"
        return self.original_output(root, *args)

    def test_create_commits_only_version_notes_and_pushes_annotated_tag_atomically(self):
        with patch.object(release, "output", side_effect=self.output), contextlib.redirect_stdout(io.StringIO()):
            release.create(self.root, "v0.1.1", self.notes)
        self.assertEqual(read_version(self.root), "0.1.1")
        self.assertEqual(set(self.git("diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD").splitlines()), {"VERSION", "changelog/0.1.1.md"})
        self.assertEqual(self.git("cat-file", "-t", "refs/tags/v0.1.1"), "tag")
        self.assertIn('$(not-a-command)', self.git("tag", "-l", "v0.1.1", "--format=%(contents)"))
        remote = self.git("ls-remote", "origin", "refs/heads/main", "refs/tags/v0.1.1^{}")
        self.assertEqual({line.split()[0] for line in remote.splitlines()}, {self.git("rev-parse", "HEAD")})
        release.validate(self.root, "v0.1.1")
        self.assertEqual(self.git("status", "--porcelain"), "")

    def test_dry_run_does_not_write_commit_tag_or_push(self):
        head = self.git("rev-parse", "HEAD")
        with patch.object(release, "output", side_effect=self.output), contextlib.redirect_stdout(io.StringIO()):
            release.create(self.root, "0.1.1", self.notes, dry_run=True)
        self.assertEqual(self.git("rev-parse", "HEAD"), head)
        self.assertEqual(read_version(self.root), "0.1.0")
        self.assertEqual(self.git("tag", "--list"), "")
        self.assertEqual(self.git("status", "--porcelain"), "")

    def test_dirty_unpushed_and_empty_notes_fail_before_version_mutation(self):
        dirty = self.root / "untracked.cpp"
        dirty.touch()
        with self.assertRaisesRegex(ValueError, "dirty"):
            release.new_version(self.root, "0.1.1", self.notes)
        dirty.unlink()
        self.notes.write_text(" \n")
        with self.assertRaisesRegex(ValueError, "empty"):
            release.new_version(self.root, "0.1.1", self.notes)
        self.notes.write_text("New feature")
        (self.root / "VERSION").write_text("0.1.1\n")
        self.git("commit", "-am", "Unpushed source")
        with patch.object(release, "output", side_effect=self.output):
            with self.assertRaisesRegex(ValueError, "push the reviewed"):
                release.create(self.root, "0.1.2", self.notes)
        self.assertEqual(read_version(self.root), "0.1.1")
        self.assertFalse((self.root / "changelog/0.1.2.md").exists())

    def test_validation_rejects_lightweight_wrong_version_and_changed_notes(self):
        release.prepare(self.root, "0.1.1", "Release notes")
        self.git("add", "VERSION", "changelog/0.1.1.md")
        self.git("commit", "-m", "Version")
        self.git("tag", "v0.1.1")
        with self.assertRaisesRegex(ValueError, "annotated"):
            release.validate(self.root, "v0.1.1")
        self.git("tag", "-d", "v0.1.1")
        self.git("tag", "-a", "v0.1.1", "-m", "Notes")
        with self.assertRaisesRegex(ValueError, "match VERSION"):
            release.validate(self.root, "v0.1.2")
        (self.root / "changelog/0.1.1.md").write_text("# v0.1.1\n\nChanged notes\n")
        with self.assertRaisesRegex(ValueError, "differ"):
            release.validate(self.root, "v0.1.1")


    def test_validate_requires_version_to_belong_to_tagged_commit(self):
        release.prepare(self.root, "0.1.1", "Release notes")
        self.git("add", "VERSION", "changelog/0.1.1.md")
        self.git("commit", "-m", "Version")
        self.git("tag", "-a", "v0.1.2", "-m", "Incorrect tag")
        (self.root / "VERSION").write_text("0.1.2\n")
        with self.assertRaisesRegex(ValueError, "VERSION differs"):
            release.validate(self.root, "v0.1.2")



class UploadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        binary = root / "binary"
        binary.write_bytes(b"server")
        self.directory = root / "assets"
        for target in TARGETS:
            package(binary, self.directory, "1.2.3", target)
        self.notes = root / "notes.md"
        self.notes.write_text("# v1.2.3\n\nNotes\n")
        self.remote = {"draft": True, "html_url": "https://github.com/example/repo/releases/tag/v1.2.3", "assets": [
            {"name": path.name, "size": path.stat().st_size, "digest": "sha256:" + checksum(path)}
            for path in self.directory.iterdir()]}
        for name, value in [("validate", ("1.2.3", self.notes)), ("github_repo", "example/repo")]:
            patcher = patch.object(release, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_upload_publishes_only_after_all_assets_verified(self):
        with patch.object(release, "release_info", side_effect=[None, self.remote, {**self.remote, "draft": False}]), patch.object(release, "output") as call, contextlib.redirect_stdout(io.StringIO()):
            release.upload(Path(), "v1.2.3", self.directory, "example/repo")
        args = [entry.args[1:] for entry in call.call_args_list]
        self.assertIn("--verify-tag", args[0])
        self.assertIn("--draft", args[0])
        self.assertEqual(args[1][:3], ("gh", "release", "upload"))
        self.assertIn("--clobber", args[1])
        self.assertEqual(args[2][-1], "--draft=false")

    def test_published_and_partial_failures_never_publish_or_overwrite(self):
        with patch.object(release, "release_info", return_value={**self.remote, "draft": False}), patch.object(release, "output") as call:
            with self.assertRaisesRegex(ValueError, "already published"):
                release.upload(Path(), "v1.2.3", self.directory, "example/repo")
            call.assert_not_called()
        incomplete = {**self.remote, "assets": self.remote["assets"][:-1]}
        with patch.object(release, "release_info", side_effect=[None, incomplete]), patch.object(release, "output") as call:
            with self.assertRaisesRegex(ValueError, "missing assets"):
                release.upload(Path(), "v1.2.3", self.directory, "example/repo")
            self.assertFalse(any("--draft=false" in entry.args for entry in call.call_args_list))

    def test_draft_retry_and_remote_digest_mismatch(self):
        wrong = {**self.remote, "assets": [{**item, "digest": "sha256:bad"} for item in self.remote["assets"]]}
        with patch.object(release, "release_info", side_effect=[self.remote, wrong]), patch.object(release, "output") as call:
            with self.assertRaisesRegex(ValueError, "verification failed"):
                release.upload(Path(), "v1.2.3", self.directory, "example/repo")
            self.assertFalse(any("create" in entry.args for entry in call.call_args_list))
            self.assertFalse(any("--draft=false" in entry.args for entry in call.call_args_list))


if __name__ == "__main__":
    unittest.main()
