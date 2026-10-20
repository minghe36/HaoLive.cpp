"""Verify all embedded notices survive compiler and packaging changes."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
notices = ["LICENSE", "THIRD_PARTY_NOTICES.md", "third_party/ggml/LICENSE",
           "third_party/ixwebsocket/LICENSE", "third_party/licenses/httplib.txt",
           "third_party/licenses/json.txt", "third_party/licenses/moonshine.txt"]
expected = "".join("\n" + name + "\n\n" + (root / name).read_text(encoding="utf-8") + "\n" for name in notices)
actual = subprocess.check_output([sys.argv[1], "--licenses"]).decode("utf-8").replace("\r\n", "\n")
assert actual == expected, "embedded license text differs from source notices"
