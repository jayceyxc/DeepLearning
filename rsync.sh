#!/bin/sh

rsync -aP --exclude-from="exclude.txt" . root@lab_tf_2.0:/workspace/yuxuecheng/tf_workspace/DeepLearning
