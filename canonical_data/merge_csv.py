#!/usr/bin/python
import sys

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Must have at least 2 args")
        sys.exit(1)

    srcs, target = sys.argv[1:-1], sys.argv[-1]
    
    lines = []
    for srcI,src in enumerate(srcs):
        data = open(src).read().strip().split("\n")
        lines += data if srcI == 0 else data[1:]
    
    print len(lines)
    open(target,'w').write("\n".join(lines))

