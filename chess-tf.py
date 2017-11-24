import tensorflow as tf
import numpy

def autoEncoder(netInput,netOutput):
    
    conv1 = tf.layers.conv2d(
      inputs=netInput,
      filters=6,
      kernel_size=[12,1,1],
      padding="same",
      activation=tf.nn.relu)
    conv1Flat = tf.reshape(conv1, [-1])
    dense = tf.layers.dense(inputs=conv1Flat, shape =[768], activation=tf.nn.relu)
    reshape = conv1Flat = tf.reshape(conv1, [8,8,12], name = 'outputTensor')

    loss = tf.losses.mean_squared_error(netOutput,reshape)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    
    return tf.estimator.EstimatorSpec(loss=loss, train_op=train_op)


def main():
    noise = tf.random_normal(
    shape=[12,8,8],
    mean=5.0,
    stddev=1.0,
    dtype=tf.float32,
    seed=None,
    name=None)

    autoEncoderEstimator = tf.estimator.Estimator(model_fn=autoEncoder, model_dir="/tmp/mnist_convnet_model")

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": noise},
    y=noise,
    batch_size=1,
    num_epochs=None,
    shuffle=False)

    autoEncoderEstimator.train(
        input_fn=train_input_fn,
        steps=20000)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": noise},
        y=noise,
        num_epochs=1,
        shuffle=False)
    eval_results = autoEncoderEstimator.evaluate(input_fn=eval_input_fn)
    print(eval_results)

