enviroment installation

1. $vt_handoff acpi_osi=linux nomodeset
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
12 sudo bash Anaconda.sh(yes to prepend)
13. sudo chown -R shuoyuan /home/shuoyuan/anaconda3 
    sudo chmod -R +x /home/shuoyuan/anaconda3
14. 
    anaconda search -t conda tensorflow
    anaconda show ???????/tensorflow
    conda install --channel https:?????????????
    
    
            test
            import tensorflow as tf
            hello = tf.constant('Hello, TensorFlow!')
            sess = tf.Session()
            print(sess.run(hello))
15. 
    conda install --channel https://conda.anaconda.org/anaconda opencv
    
    
https://cranfield.wless.cranfield.ac.uk/guest/guest-template_2.php?portal_ip=138.250.54.3&client_id=3c:a0:67:4a:9a:72&wbaredirect=http://www.baidu.com/&ssid=Cranfield%20Web&bssid=3c:94:d5:78:01:42&_browser=1


http://blog.csdn.net/u011596455/article/details/53240221


http://www.cnblogs.com/mywolrd/p/4756005.html
