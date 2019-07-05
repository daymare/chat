

import subprocess
import os
import sys


def tee_output(output_dir, output_prefix):
    # ensure output dir exists
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # ensure a unique filename
    postfix_no = 0
    base_name = os.path.join(output_dir, output_prefix)
    fname = base_name + ".txt"

    while os.path.isfile(fname) is True:
        fname = "%s-%02d.txt" % (base_name, postfix_no)
        postfix_no += 1

    # tee stdout and stderr to filename
    tee = subprocess.Popen(["tee", fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

