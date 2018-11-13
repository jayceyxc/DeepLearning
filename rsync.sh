#!/bin/sh

rsync -aP --exclude-from="exclude.txt" . yuxuecheng@tf:/home/yuxuecheng/
