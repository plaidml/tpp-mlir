#!/usr/bin/env python3

import os
import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument('filename')

def syntax():
    print("Use -h for options\n")
    exit(1)

if __name__ == '__main__':
    args = argParser.parse_args()
    if args.filename is None:
        syntax()

    if not os.path.isfile(args.filename):
        print(f"{args.filename} is not a file\n")
        syntax()

    ext=os.path.splitext(args.filename)[1]
    counter = 0
    filename = f"%03d{ext}" % counter
    outfile = open(filename, "a")
    with open(args.filename, "r") as file:
        for line in file:
            # New files
            if line.startswith("//"):
                counter += 1
                filename = f"%03d{ext}" % counter
                outfile.close()
                outfile = open(filename, "w")
            # Write to outfile
            outfile.write(line)
    outfile.close()
