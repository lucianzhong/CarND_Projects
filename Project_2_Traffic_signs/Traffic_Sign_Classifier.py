import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import pickle

# read in the training,validation and testing data
training_file = "./traffic-signs-data/train.p"
validation_file = "./traffic-signs-data/valid.p"
testing_file = "./traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

# read in the properties of the traffic signs data sets
n_train = len(X_train)
n_valid = len(X_valid)
n_test = len(X_test)
image_shape = X_train.shape[1:]
labels_pd = pd.read_csv("./signnames.csv")
n_classes = len(labels_pd)

# print the properties of the traffic signs data sets
print("Number of training examples =", n_train)
print("Number of validation examples =", n_valid)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


# plot the training images
fig, axs = plt.subplots(2,4, figsize=(20, 10))
axs = axs.ravel()
labels_sign = labels_pd.T.to_dict()
for i in range(8):
    index = random.randint(0, len(X_train))
    image = X_train[index]
    axs[i].axis('off')
    axs[i].imshow(image)
    axs[i].set_title(str(y_train[index]) + ":" + labels_sign[y_train[index]]['SignName'])

# distribution of classes in train,validation and test data
height_train, bins_train = np.histogram(y_train, bins=n_classes)
height_train = height_train/n_train
wist_train = (bins_train[:-1] + bins_train[1:]) / 2

height_valid, bins_valid = np.histogram(y_valid, bins=n_classes)
height_valid = height_valid/n_valid
wist_valid = (bins_valid[:-1] + bins_valid[1:]) / 2

height_test, bins_test = np.histogram(y_test, bins=n_classes)
height_test = height_test/n_test
wist_test = (bins_test[:-1] + bins_test[1:]) / 2

plt.subplots(figsize=(20, 5))
plt.subplot(1,3,1)
plt.bar(wist_train,height_train)
plt.title('distribution of classes in the training')
plt.subplot(1,3,2)
plt.bar(wist_valid,height_valid)
plt.title('distribution of classes in the validation')
plt.subplot(1,3,3)
plt.bar(wist_test,height_test)
plt.title('distribution of classes in the test')

plt.show()

# Pre-processing training data with grayscale and normalization
# grayscale
X_train = np.sum(X_train/3, axis=3, keepdims=True)
X_valid = np.sum(X_valid/3, axis=3, keepdims=True)
X_test = np.sum(X_test/3, axis=3, keepdims=True)

# Normalization
X_train = (X_train - 128)/128
X_valid = (X_valid - 128)/128
X_test = (X_test - 128)/128

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('X_valid shape:', X_valid.shape)

# Shuffle the training data
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten

# define the initial architecture of the LeNet
def LeNet(x):
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional layer  Input shape:32x32x1  Output shape:28x28x6
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Relu activation
    conv1 = tf.nn.relu(conv1)

    # Pooling  Input shape:28x28x6  Output shape:14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional layer Output shape:10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # Relu activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input shape:10x10x16 Output shape:5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input shape:5x5x16 Output shape:400.
    fc0 = flatten(conv2)

    # Layer 3: Fully connected. Input shape:400  Output shape:120
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # Relu activation
    fc1 = tf.nn.relu(fc1)

    # Layer 4: Fully connected  Input shape:120 Output shape:84
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # Relu activation.
    fc2 = tf.nn.relu(fc2)

    # Layer 5: Fully Connected  Input shape:84 Output shape 43.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, 43), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits

# Modify the LeNet to improve accuracy
def LeNet_modified(x):
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional layer  Input shape:32x32x1  Output shape:28x28x6
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Relu activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input shape:28x28x6. Output shape:14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional Output shape:10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # Relu activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input shape:10x10x16. Output shape:5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 3: Convolutional. Output shape:1x1x400.
    conv3_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 16, 400), mean=mu, stddev=sigma))
    conv3_b = tf.Variable(tf.zeros(400))
    conv3 = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='VALID') + conv3_b

    #Relu activation
    conv3 = tf.nn.relu(conv3)

    # Flatten  Input shape:1x1x400. Output shape:400
    conv3_flat = flatten(conv3)


    #Flatten Input shape:5x5x16. Output shape:400.
    fc0 = flatten(conv2)

    # Concat Input shape: 400 and 400. Output shape:800
    fc0 = tf.concat(1, [conv3_flat, fc0])

    # Dropout
    fc0 = tf.nn.dropout(fc0, keep_prob)

    # Layer 4: Fully Connected  Input shape:800. Output shape:43.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(800, 43), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc0, fc1_W) + fc1_b
    return logits

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)
keep_prob = tf.placeholder(tf.float32)

# Train LeNet and calculate the accuracy based on training data and validation data
# Hyperparameters
EPOCHS = 80
BATCH_SIZE = 128
rate = 0.0008
dropout_train = 0.7
dropout_valid = 1.0

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

# Define accuracy calculation function
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# Training LetNet in every epoch
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        # Calculation validation accuracy
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './lenet_modified')
    print("Model saved")

# Calculation test set accuracy
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('./lenet_modified.meta')
    saver.restore(sess, "./lenet_modified")
    my_accuracy = evaluate(X_test, y_test)
    print("Test Set Accuracy = {:.3f}".format(my_accuracy))

#Load the German_Traffic_signs images
import glob
import cv2

fig.axs = plt.subplots(1, 5, figsize=(18, 2))
fig.subplots_adjust(hspace=.2, wspace=.001)
axs = axs.ravel()
new_images = []

for i, img in enumerate(glob.glob('./German_traffic_signs/image*.png')):
    image = cv2.imread(img)
    image = cv2.resize(image, (32, 32))
    axs[i].axis('off')
    axs[i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = np.reshape(np.asarray(image), [32, 32, 3])
    new_images.append(image)

new_images = np.asarray(new_images)
new_images_gry = np.sum(new_images / 3, axis=3, keepdims=True)
new_images_normalized = (new_images_gry - 128) / 128
print(new_images_normalized.shape)

# Using the trained LeNet network to classify German_Trafic_signs

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('./lenet_modified.meta')
    saver.restore(sess, "./lenet_modified")
    prediction = tf.nn.softmax(logits)
    classification = sess.run(prediction, feed_dict={x: new_images_normalized, keep_prob: dropout_valid})
    result = np.array([])

for i in range(5):
    print(np.argmax(classification[i]))
    result = np.append(result, np.argmax(classification[i]))

# Output Top 5 Softmax Probabilities For Each Image
fig, axs = plt.subplots(5, 2, figsize=(10, 20))
axs = axs.ravel()
with tf.Session() as sess:
    for i in range(len(classification)):
        print(sess.run(tf.nn.top_k(tf.constant(classification[i]), k=5)))

