# object-detection

Caffe with SSD CPU Only Compile 


ImportError: libcaffe.so.1.0.0-rc3: cannot open shared object file: No such file or directory



http://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/project.html#datasets

https://www.dropbox.com/s/tjvh43e0vrbki05/TownCentreXVID.mp4?dl=0


ftp://ftp.cs.rdg.ac.uk/pub/PETS2009/Crowd_PETS09_dataset/a_data/S2_L2/


Server terminated abruptly (error code: 14, error message: '', log file: '/home/jinghua/.cache/bazel/_bazel_root/fae3fa6420780455bb10cce6b449c4e0/server/jvm.out')
https://zhuanlan.zhihu.com/p/27168325?utm_source=wechat_session&utm_medium=social



enviroment installation

1. 
2. sudo apt-get update
3. sudo gedit /etc/modprobe.d/blacklist.conf
    blacklist nouveau
    options nouveau modset=0
4. sudo update-initramfs -u
    check if 
      lspci | grep nouveau
5. ctrl+alt+f1
6. sudo service lightdm stop
7. sudo chmod 755 NVIDIA.run
8. sudo ./NVIDIA.run --no-opengl-files
9. sudo service lightdm start
10. ctrl+alt+f7
11. nvidia-smi
12 sudo bash Anaconda.sh
