# face_recognition
A face_recognition program with tensorflow and keras

#requirements

1. 安装pyenv以及virtualenv搭建虚拟环境

   1. 安装下载工具curl
      ​    sudo apt-get install curl

   2. 使用curl下载pyenv
      curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
      会出现提示
      export PATH="~/.pyenv/bin:$PATH"
      eval "$(pyenv init -)"
      eval "$(pyenv virtualenv-init -)"
      这是指你的pyenv没有加到环境变量中

   3. 将pyenv添加到环境变量中
      sudo vim ~/.bashrc
      将上述提示内容加入到末尾即可

   4. 使配置文件生效
      source ~/.bashrc

   5. 检查是否安装成功
      终端输入： pyenv 看有无输出

   6. 利用pyenv搭建虚拟python环境

      pyenv install 3.6.5                 #安装python 3.6.5

      pyenv virtualenv 3.6.5 ML36            #新建virtualenv环境(使用ML36作为环境名时，可以在进入face_recognition项目目录时自动启动环境)

      pyenv activate ML36                   #启动环境

      pip install opencv-python tensorflow-gpu keras torch numpy scikit-learn matplotlib

   7. 安装过程可能遇到的错误

      1. import numpy 和 import cv2时出错

         这是由于安装了多个版本的numpy导致，初开用户安装的numpy之外，opencv也会安装numpy；将多余的numpy卸载后便可以解决这个问题

      2. import matplotlib出错

         如果是提醒pip版本过低出错，需要升级pip，在升级之后需要修改pip的原始文件，不然会出现“ImportError: cannot import name main”的错误

         请参考如下链接进行修改：https://www.jianshu.com/p/7ed03300408f

         如果是出现_tkinter错误，则需要在安装python3.6.5时，需要添加tkinter支持

2. 数据集的准备

   在home/username/Documents文件夹下存放数据集，数据集文件夹结构如下： ./DataBase/student_id/images (只需要脸集)

   修改项目目录下scripts/for_dataset/rename_images.py 中的DATA_BASE为上述数据集存放的绝对路径，然后运行，将每个类别下的图片从10000.png开始命名

   如果使用keras框架，则已经完成了数据集的基本准备，注意修改网络最后一层的神经元数为所使用训练集的类别数

   如果要使用pytorch框架，在修改路径之后运行 ./scripts/for_dataset/move_file.py 运行之后会得到手动划分的比例为7：3的训练数据和验证数据

   使用 ./scripts/for_dataset/create_test_data.py 来获得每个类别50张的测试数据，来对训练好的每模型进行测试

#Usage

1. Keras & Tensorflow 框架

   1. 程序运行(确保在face_recognition项目目录下)

      cd ./keras/src

      python main.py

      然后按照控制台中提示输入即可

   2. 实现功能

      基于tensorflow框架的数据读取和储存

      使用DataShuffleSplit，Kflod两种功能进行训练集的划分，并进行训练

      使用训练好的模型对摄像头中的实时视频流或者批量测试图片进行预测

   3. 待完善功能

      完善tensorflow版本中的训练部分，解决模型训练时的异常

      完善Keras版本中的GridSearch部分，尝试进行epoch， batch_size，optimizer等超参数的训练

2. Pytorch  框架

   1. 程序运行(确保在face_recognition项目目录下)

      1. 训练模型

         cd ./pytorch/src

         python train.py

      2. 测试模型

         cd ./pytorch/src

         python detect.py

   2. 实现功能

      基于pytorch框架的数据读取，储存

      使用pytorch自带的api进行模型的训练并保存

      使用训练好的模型对测试图片进行预测

   3. 待实现功能

      对于实时视频流的检测

      代码进一步整合和模块化

