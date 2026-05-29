#!/usr/bin/env python3
import os
import subprocess


def main():
    os.chdir(r"rsciio/tests")
    cmd = ("python", "registry_utils.py")

    return subprocess.call(cmd)


if __name__ == "__main__":
    # Script used by the pre-commit hook
    exit(main())
