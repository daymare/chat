#!/bin/bash


lsof -ti:6006 | xargs kill

tensorboard \
    --logdir=train/ \
    --port 6006
#    --debugger_port 6064



