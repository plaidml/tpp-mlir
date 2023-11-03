#!/usr/bin/env python3

import os
import argparse
import subprocess

argParser = argparse.ArgumentParser()
argParser.add_argument("-p", "--path")
argParser.add_argument("-d", "--diff")
argParser.add_argument("ext")


def syntax():
    print("Use -h for options\n")
    exit(1)


def getFileName(counter):
    return f"{args.path}/%03d.{args.ext}" % counter


def readFile(file):
    lines = list()
    with open(file, "r") as fp:
        for line in fp:
            if line is None or line == "":
                continue
            if line.startswith("//"):
                continue
            lines.append(line)
    return lines


def sameFile(prev, next):
    lhs = readFile(prev)
    rhs = readFile(next)
    return lhs == rhs


def nextFile(counter):
    counter += 1
    file = getFileName(counter)
    while os.path.isfile(file):
        with open(file, "r") as fp:
            if len(fp.readlines()) > 3:
                return file, counter
        counter += 1
        file = getFileName(counter)
    return None, counter


if __name__ == "__main__":
    args = argParser.parse_args()
    if args.ext is None:
        syntax()

    if args.path is None:
        args.path = "."

    if not os.path.isdir(args.path):
        print(f"{args.path} not a directory\n")
        syntax()

    if not os.path.isfile(f"{args.path}/000.{args.ext}"):
        print(f"000.{args.ext} is not a file\n")
        print(f"Run 'split.py` first?\n")
        syntax()

    if not os.path.isfile(f"{args.path}/001.{args.ext}"):
        print(f"001.{args.ext} is not a file\n")
        print(f"Needs at least two files to compare\n")
        print(f"Run 'split.py` first?\n")
        syntax()

    if args.diff is None:
        args.diff = "diff"

    curr, counter = nextFile(0)
    last = curr
    while True:
        next, counter = nextFile(counter)
        if next is None:
            print("No more files\n")
            break
        if sameFile(curr, next):
            curr = next
            continue
        cmd = [args.diff, last, next]
        print(" ".join(cmd))
        print("Press ENTER to continue or CTRL+C to cancel...\n")
        input()
        subprocess.run(cmd)
        curr = next
        last = curr
