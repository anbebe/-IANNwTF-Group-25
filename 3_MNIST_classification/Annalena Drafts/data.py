import tensorflow_datasets as tfds
import tensorflow as tf


def load_data():
    '''
    MNIST dataset: 10.000 test images, 60.000 train images with pixel format (28,28,1)
    pixel values between 0 and 255 uint8 (grey-scale)
    label values: 10
    '''
    (train_ds, test_ds), ds_info = tfds.load('mnist', split =['train', 'test'], as_supervised=True, with_info=True)
    #print(ds_info)
    #tfds.show_examples(train_ds, ds_info)
    return train_ds, test_ds

def preprocess_data(data, batch_size):
    data = data.map(lambda x, t: (tf.cast(x, float), t))
    data = data.map(lambda x, t: (tf.reshape(x, (-1,)), t))
    data = data.map(lambda x, t: ((x/128.)-1., t))
    data = data.map(lambda x, t: (x, tf.one_hot(t, depth=10)))
    data = data.cache()
    data = data.shuffle(1000)
    data = data.batch(batch_size)
    data = data.prefetch(20)
    return data
    
