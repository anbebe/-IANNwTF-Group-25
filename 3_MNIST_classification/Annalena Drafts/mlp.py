import tensorflow as tf
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from data import load_data, preprocess_data

class MNISTModel(tf.keras.Model):

  def __init__(self):
    super().__init__()

    #optimizer, loss function and metrics
    #use categorical accuracy due to one hot encoding labels
    self.metrics_list = [tf.keras.metrics.CategoricalCrossentropy(name="loss"), 
                        tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
    self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    self.loss = tf.keras.losses.CategoricalCrossentropy()

    self.dense1 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
    self.dense2 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
    # return categorical probability distributon instead of logits via softmax
    self.out_layer = tf.keras.layers.Dense(10, activation=tf.nn.softmax)

  @tf.function
  def call(self, image, training=False):
    # forward computation
    out = self.dense1(image)
    out = self.dense2(out)
    return self.out_layer(out)

  @property
  def metrics(self):
    # return a list with all metrics in the model
    return self.metrics_list


  def reset_metrics(self):
    for metric in self.metrics_list:
      metric.reset_states()

  @tf.function
  def train_step(self, data):
    # update the state of the metrics according to loss
    # return a dictionary with metrics name as keys an metric results
    img, label = data
    with tf.GradientTape() as tape:
      output = self(img, training=True)
      loss = self.loss(label, output) + tf.reduce_sum(self.losses)

    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    self.metrics_list[0].update_state(label, output)
    self.metrics_list[1].update_state(label, output)

    return [ m.result() for m in self.metrics_list]


  @tf.function
  def test_step(self, data):
    img, label = data
    output = self(img, training=False)    
    loss = self.loss(label, output) + tf.reduce_sum(self.losses)
    self.metrics_list[0].update_state(label, output)
    self.metrics_list[1].update_state(label, output)
    return [ m.result() for m in self.metrics_list]


def train(epochs, model, train_ds, test_ds):
    metric_list = []
    for epoch in range(epochs):
      print("Epoch ", epoch)

      metrics = []

      for sample in tqdm.tqdm(train_ds, position=0, leave=True):
        metrics.append(model.train_step(sample))
      print(np.asarray(metrics).shape)
      mean_metrics = np.mean(np.asarray(metrics),axis=1)
      print(f"loss: {mean_metrics[0]}, accuracy: {mean_metrics[1]}")
      model.reset_metrics()

      # Validation
      for sample in test_ds:
        metrics.append(model.test_step(sample))
      val_mean_metrics = np.mean(np.asarray(metrics),axis=1)
      print(f"val_loss: {val_mean_metrics[0]}, val_accuracy: {val_mean_metrics[1]}")
      model.reset_metrics()

      metric_list.append([mean_metrics[0], mean_metrics[1], val_mean_metrics[0], val_mean_metrics[1]])

    return metric_list

def visualisation(train_losses, train_acc, test_losses, test_acc):
  plt.figure ()
  line1, = plt.plot(train_losses , "b-" )
  line2, = plt.plot(test_losses , "r-" )
  line3, = plt.plot(train_acc , "b:" )
  line4, = plt.plot(test_acc , "r:" )
  plt.xlabel( " Training steps " )
  plt.ylabel( " Loss / Accuracy " )
  plt.legend((line1, line2, line3, line4), ("training loss", "test loss", "train accuracy", "test accuracy"))
  plt.show()
  


if __name__ == "__main__":
  train_ds, test_ds = load_data()
  train_ds = preprocess_data(train_ds, batch_size=32)
  test_ds = preprocess_data(test_ds, batch_size=32)

  mnist_model = MNISTModel()
  #mnist_model(tf.keras.Input((28,28,1)))
  #mnist_model.summary()

  metrics = train(epochs=10, model=mnist_model, train_ds=train_ds, test_ds=test_ds)
  metrics = np.asarray(metrics)
  visualisation(metrics[:,0], metrics[:,1], metrics[:,2], metrics[:,2])