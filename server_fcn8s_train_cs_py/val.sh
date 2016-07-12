
LOG=log/val-30k-`date +%Y-%m-%d-%H-%M-%S`.log

caffe test -model val8s.prototxt -gpu 1 -weights ~/jacques/fcn_cs_more_accurate/server_fcn8s_train_cs_py/snapshot/train_iter_30000.caffemodel -iterations 500 2>&1 | tee $LOG



LOG=log/val-56k-`date +%Y-%m-%d-%H-%M-%S`.log

caffe test -model val8s.prototxt -gpu 1 -weights ~/jacques/fcn_cs/server_fcn8s_train_cs_py/snapshot/train_iter_56000.caffemodel -iterations 500 2>&1 | tee $LOG
