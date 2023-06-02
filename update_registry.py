#!/usr/bin/env python3
import os
import subprocess
import sys


def main():
    os.chdir(r"rsciio/tests")
    print(os.listdir())
    cmd = ("python", "registry_utils.py", *sys.argv[1:])

    return subprocess.call(cmd)


if __name__ == "__main__":
    # Script used by the pre-commit hook
    exit(main())
