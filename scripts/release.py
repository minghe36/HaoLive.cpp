#!/usr/bin/env python3
"""Prepare versions, push annotated tags, and publish tested CI artifacts."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

from release_common import checksum, read_version, release_assets, version

ROOT = Path(__file__).resolve().parents[1]


def output(root, *args):
    return subprocess.check_output(list(args), cwd=root, text=True, encoding="utf-8").strip()


def clean(root):
    if output(root, "git", "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("working tree is dirty; review and commit source changes before releasing")


def github_repo(root):
    remote = output(root, "git", "remote", "get-url", "origin")
    match = re.fullmatch(r"(?:git@github\.com:|https://github\.com/)([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?", remote)
    if not match:
        raise ValueError("origin must be a GitHub SSH or HTTPS repository URL")
    return match[1]


def notes_path(root, number):
    return Path(root) / "changelog" / (version(number) + ".md")


def new_version(root, requested, notes_file):
    number = version(requested)
    clean(root)
    if tuple(map(int, number.split("."))) <= tuple(map(int, read_version(root).split("."))):
        raise ValueError("new version must be greater than VERSION")
    notes = Path(notes_file).read_text(encoding="utf-8-sig").strip()
    if not notes:
        raise ValueError("release notes must not be empty")
    if notes_path(root, number).exists():
        raise ValueError("changelog already exists")
    if output(root, "git", "tag", "--list", "v" + number):
        raise ValueError("local release tag already exists")
    return number, notes


def prepare(root, number, notes):
    path = notes_path(root, number)
    path.parent.mkdir(exist_ok=True)
    # The changelog is created first, so an existing file cannot be overwritten.
    with path.open("x", encoding="utf-8", newline="\n") as target:
        target.write(f"# v{number}\n\n{notes}\n")
    (Path(root) / "VERSION").write_text(number + "\n", encoding="utf-8", newline="\n")


def remote_preflight(root, number):
    repo = github_repo(root)
    branch = output(root, "git", "symbolic-ref", "--quiet", "--short", "HEAD")
    if not branch:
        raise ValueError("release from a named branch")
    current = output(root, "git", "rev-parse", "HEAD")
    remote = output(root, "git", "ls-remote", "origin", "refs/heads/" + branch)
    if not remote or remote.split()[0] != current:
        raise ValueError("push the reviewed source commit to origin before creating a release")
    if output(root, "git", "ls-remote", "origin", "refs/tags/v" + number):
        raise ValueError("remote tag already exists; rerun its workflow instead of moving the tag")
    output(root, "gh", "api", "repos/" + repo, "--jq", ".full_name")
    return branch, repo


def validate(root, tag):
    number = read_version(root)
    if tag != "v" + number:
        raise ValueError("tag does not match VERSION")
    if output(root, "git", "show", "HEAD:VERSION") != number:
        raise ValueError("VERSION differs from the tagged commit")
    if output(root, "git", "cat-file", "-t", "refs/tags/" + tag) != "tag":
        raise ValueError("release tag must be annotated")
    if output(root, "git", "rev-parse", tag + "^{commit}") != output(root, "git", "rev-parse", "HEAD"):
        raise ValueError("checkout must point to the release tag commit")
    path = notes_path(root, number)
    text = path.read_text(encoding="utf-8").strip()
    heading, separator, body = text.partition("\n")
    if heading != "# " + tag or not separator or not body.strip():
        raise ValueError("changelog must contain its version heading and nonempty notes")
    tracked = output(root, "git", "show", "HEAD:changelog/" + number + ".md")
    if tracked != text:
        raise ValueError("release notes differ from the tagged commit")
    return number, path


def create(root, requested, notes_file, dry_run=False):
    number, notes = new_version(root, requested, notes_file)
    branch, repo = remote_preflight(root, number)
    # A missing workflow must fail before any local version/commit/tag mutation.
    output(root, "git", "ls-files", "--error-unmatch", ".github/workflows/release.yml", "scripts/build.py", "scripts/release.py", "VERSION")
    if dry_run:
        print(f"Would create v{number}, commit VERSION/changelog, atomically push {branch} + tag to {repo}.")
        return
    prepare(root, number, notes)
    output(root, "git", "add", "--", "VERSION", "changelog/" + number + ".md")
    output(root, "git", "commit", "-m", "Release v" + number, "--", "VERSION", "changelog/" + number + ".md")
    clean(root)
    output(root, "git", "tag", "-a", "v" + number, "-F", str(notes_path(root, number)))
    validate(root, "v" + number)
    output(root, "git", "push", "--atomic", "origin", "HEAD:refs/heads/" + branch, "refs/tags/v" + number)
    print(f"Pushed v{number}. Build/upload status: https://github.com/{repo}/actions/workflows/release.yml")


def push_tag(root, dry_run=False):
    clean(root)
    number = read_version(root)
    tag = "v" + number
    if output(root, "git", "show", "HEAD:VERSION") != number:
        raise ValueError("VERSION is not committed")
    _, repo = remote_preflight(root, number)
    if output(root, "git", "tag", "--list", tag):
        validate(root, tag)
    else:
        path = notes_path(root, number)
        text = path.read_text(encoding="utf-8").strip()
        if not text.startswith("# " + tag + "\n") or not text.partition("\n")[2].strip():
            raise ValueError("missing versioned release notes")
        if output(root, "git", "show", "HEAD:changelog/" + number + ".md") != text:
            raise ValueError("release notes are not committed")
    if dry_run:
        print(f"Would push annotated tag {tag} to {repo}.")
        return
    if not output(root, "git", "tag", "--list", tag):
        output(root, "git", "tag", "-a", tag, "-F", str(notes_path(root, number)))
    output(root, "git", "push", "origin", "refs/tags/" + tag)
    print(f"Pushed {tag}. Build/upload status: https://github.com/{repo}/actions/workflows/release.yml")


def release_info(root, repo, tag):
    text = output(root, "gh", "api", "--paginate", f"repos/{repo}/releases?per_page=100",
                  "--jq", '.[] | select(.tag_name == "' + tag + '")')
    return json.loads(text) if text else None


def upload(root, tag, directory, repo):
    number, notes = validate(root, tag)
    if repo != github_repo(root):
        raise ValueError("release repository does not match origin")
    assets = release_assets(directory, number)
    existing = release_info(root, repo, tag)
    if existing and not existing["draft"]:
        raise ValueError("release is already published; published assets are never overwritten")
    names = {path.name for path in assets}
    if existing and {item["name"] for item in existing["assets"]} - names:
        raise ValueError("draft contains unexpected assets")
    if not existing:
        output(root, "gh", "release", "create", tag, "--repo", repo, "--verify-tag", "--draft",
               "--title", tag, "--notes-file", str(notes))
    else:
        output(root, "gh", "release", "edit", tag, "--repo", repo, "--title", tag, "--notes-file", str(notes))
    # Clobber is restricted to a draft, so retries cannot replace published files.
    output(root, "gh", "release", "upload", tag, *map(str, assets), "--repo", repo, "--clobber")
    uploaded = release_info(root, repo, tag)
    if not uploaded or not uploaded["draft"]:
        raise ValueError("draft disappeared or was externally published during upload")
    remote = {asset["name"]: asset for asset in uploaded["assets"]}
    if set(remote) != names:
        raise ValueError("remote release is missing assets")
    for path in assets:
        asset = remote[path.name]
        if asset["size"] != path.stat().st_size or (asset.get("digest") and asset["digest"] != "sha256:" + checksum(path)):
            raise ValueError("remote asset verification failed: " + path.name)
    output(root, "gh", "release", "edit", tag, "--repo", repo, "--draft=false")
    published = release_info(root, repo, tag)
    if not published or published["draft"]:
        raise ValueError("release did not become public")
    print(published["html_url"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "create"):
        command = commands.add_parser(name)
        command.add_argument("version")
        command.add_argument("--notes-file", type=Path, required=True)
        command.add_argument("--dry-run", action="store_true")
    publish = commands.add_parser("publish")
    publish.add_argument("--dry-run", action="store_true")
    check = commands.add_parser("validate")
    check.add_argument("--tag", required=True)
    send = commands.add_parser("upload")
    send.add_argument("--tag", required=True)
    send.add_argument("--artifacts", type=Path, required=True)
    send.add_argument("--repo", required=True)
    args = parser.parse_args(argv)
    if args.command == "create":
        create(ROOT, args.version, args.notes_file, args.dry_run)
    elif args.command == "prepare":
        number, notes = new_version(ROOT, args.version, args.notes_file)
        if args.dry_run:
            print(f"Would write VERSION={number} and changelog/{number}.md")
        else:
            prepare(ROOT, number, notes)
            print(f"Prepared v{number}; review and commit VERSION + changelog/{number}.md")
    elif args.command == "publish":
        push_tag(ROOT, args.dry_run)
    elif args.command == "validate":
        validate(ROOT, args.tag)
    else:
        upload(ROOT, args.tag, args.artifacts, args.repo)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        sys.exit("Release failed: " + str(error))
