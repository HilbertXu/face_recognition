#This is a rebuilt version for face_recognition system using Tensorflow rather than Keras
#Firstly use VGG-16
#Will try to use faceNet later


#tensorflow 中图变量
##定义图变量的方法：
1. tf.Variable.init(
    initial_value, trainable=True, collections=None, validate_shape=True, name=None
    )

initial_value------所有可转换为Tensor的类型------变量的初始值(！！！为使用tf.Variable时的必需参数)
trainable------bool------只有为True时，才会加入到GraphKeys.TRAINABLE_VARIABLES，才能对它使用Optimizer
collections------list------指定该图变量的类型，默认为[GraphKeys.GLOBAL_VARIABLES]
validate_shape------bool------若为False则不进行类型和维度检查
name------string------变量名称，若缺省则由系统自动分配一个唯一的值

使用tf.Variable()来定义时，可以定义“同名”变量，实际上系统给这两个变量还是分配了不同的名称
In [1]: import tensorflow as tf
In [2]: with tf.variable_scope('scope'):
   ...:     v1 = tf.Variable(1, name='var')
   ...:     v2 = tf.Variable(2, name='var')
   ...:
In [3]: v1.name, v2.name
Out[3]: ('scope/var:0', 'scope/var_1:0')

2. tf.get_variable(
    name，shape，dtype，initializer,trainable
)
必需参数为 name即图变量的名称

tf.get_variable的用法更加丰富，当指定名称的图变量已经存在时表示获取它，当不存在时表示定义它

为了方便共享变量，必须和reuse，tf.variable_scope()配合使用

对于tf.get_variable()若需要定义同名变量，则可以考虑新加入一层scope
In [1]: import tensorflow as tf
In [2]: with tf.variable_scope('scope1'):
   ...:     v1 = tf.get_variable('var', shape=[1])
   ...:     with tf.variable_scope('scope2'):
   ...:         v2 = tf.get_variable('var', shape=[1])
   ...:
In [3]: v1.name, v2.name
Out[3]: ('scope1/var:0', 'scope1/scope2/var:0')




3. tf.placehold(dtype, shape)
主要用于真实输入数据和输出标签的输入，用于在feed_dict中的变量，不需要指定初始值，具体值在feed_dict中的变量给出
'
images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_PIXELS))
labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))

＃This is how you feed the training examples during the training:

for step in xrange(FLAGS.max_steps):
    feed_dict = {
       images_placeholder: images_feed,
       labels_placeholder: labels_feed,
     }
    _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
'


##scope划分命名空间
1. tf.name_scope
    tf.name_scope会返回一个string类型
    '
    print (tf.namescope("hello"))
    >hello/
    '
    tf.name_scope用于给op_name加上前缀

2. tf.variable_scope
    返回一个对象
    print tf.get_variable("arr1", shape=[2,10], dtype=tf.float32)
    ><tensorflow.python.ops.variable_scope.VariableScope object at 0x7fbc09959210>

简单来看 
1. 使用tf.Variable()的时候，tf.name_scope()和tf.variable_scope() 都会给 Variable 和 op 的 name属性加上前缀。 
2. 使用tf.get_variable()的时候，tf.name_scope()就不会给 tf.get_variable()创建出来的Variable加前缀。但是 tf.Variable() 创建出来的就会受到 name_scope 的影响.
