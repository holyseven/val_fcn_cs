LOG=log/val-48k-`date +%Y-%m-%d-%H-%M-%S`.log

caffe test -model val.prototxt -gpu 0 -weights ./snapshot/train_iter_48000.caffemodel -iterations 500 2>&1 | tee $LOG
