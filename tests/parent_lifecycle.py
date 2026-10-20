#!/usr/bin/env python3
"""Check port-0 readiness/authentication and exit after the POSIX host dies."""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    launcher = r'''
import json,os,subprocess,sys,time
p=subprocess.Popen([sys.argv[1],'--vad-model',sys.argv[2],'--port','0','--token','parent-test','--parent-pid',str(os.getpid())],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True)
ready=json.loads(p.stdout.readline())
print(json.dumps({'pid':p.pid,**ready}),flush=True)
time.sleep(60)
'''
    parent = subprocess.Popen([sys.executable, "-c", launcher, os.path.abspath(args.server), os.path.abspath(args.model)],
                              stdout=subprocess.PIPE, text=True)
    child = None
    try:
        ready = json.loads(parent.stdout.readline())
        child = ready["pid"]
        assert ready["event"] == "ready" and ready["api_version"] == 1 and 0 < ready["port"] <= 65535
        url = f"http://127.0.0.1:{ready['port']}/health"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(urllib.request.Request(url, headers={"Authorization": "Bearer parent-test"}), timeout=5) as response:
            assert json.load(response)["vad"]["model"] == "firered-vad"
        try:
            opener.open(url, timeout=2)
            raise AssertionError("unauthenticated health accepted")
        except urllib.error.HTTPError as error:
            assert error.code == 401
        parent.kill(); parent.wait(timeout=5)
        deadline = time.monotonic() + 5
        while True:
            try:
                opener.open(url, timeout=.5)
            except urllib.error.HTTPError:
                pass  # Still alive and enforcing auth.
            except (OSError, urllib.error.URLError):
                break
            if time.monotonic() >= deadline:
                raise AssertionError("engine survived host death")
            time.sleep(.05)
        print("Port-0 handshake, bearer authentication and parent-death cleanup passed")
    finally:
        if parent.poll() is None:
            parent.kill(); parent.wait(timeout=5)
        if child:
            try:
                os.kill(child, 15)
            except ProcessLookupError:
                pass


if __name__ == "__main__":
    main()
