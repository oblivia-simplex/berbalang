#! /usr/bin/env python3

import os
import sys

LOGS=os.path.expanduser("~/logs")

def full_path(p):
    new_path = os.path.expanduser(os.path.abspath(p))
    print(f"{p} --> {new_path}")
    return new_path

def launch(experiments, binaries, gadgets, trials):
    if os.getenv("REBUILD_IMAGE"):
        os.system("docker build -t pseudosue/berbalang .")
    experiments = full_path(experiments)
    binaries = full_path(binaries)
    gadgets = full_path(gadgets)

    cmd = f"""docker container run \\
                --mount src={LOGS},dst=/root/logs,type=bind \\
                --mount src={gadgets},dst=/root/gadgets,type=bind,readonly \\
                --mount src={binaries},dst=/root/binaries,type=bind,readonly \\
                --mount src={experiments},dst=/root/experiments,type=bind,readonly \\
                pseudosue/berbalang:latest \\
                /root/trials.sh /root/experiments {trials} {LOGS}
    """
    print(cmd)
    os.system(cmd)
    return


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(f"Usage: {sys.argv[0]} <experiments> <binaries> <gadgets> <number of trials>")
        sys.exit(1)
    launch(experiments = sys.argv[1],
            binaries = sys.argv[2],
            gadgets = sys.argv[3],
            trials = int(sys.argv[4]))


