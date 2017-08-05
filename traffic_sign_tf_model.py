from matplotlib import pyplot
import tensorflow as tf
import numpy as np
import pickle
from sklearn.utils import shuffle
import os
import time
import json
from pandas import DataFrame
from pandas.io.parsers import read_csv
from sklearn.model_selection import train_test_split
from nolearn.lasagne import BatchIterator
from collections import namedtuple
from cloudlog import CloudLog
from keras.preprocessing.image import ImageDataGenerator
from skimage import exposure

def preprocess_dataset(X, y=None):
    # Convert to grayscale, e.g. single Y channel
    # Scale features to be in [0, 1]
    # X = (X - np.mean(X, axis=(1, 2)).reshape(-1, 1, 1, 3)) / \
    #     np.std(X, axis=(1, 2)).reshape(-1, 1, 1, 3)
    X=0.299*X[:,:,:,0]+0.587*X[:,:,:,1]+0.114*X[:,:,:,2]
    X=(X/255.0).astype(np.float32)
    for i in range(len(X)):
        X[i]=exposure.equalize_adapthist(X[i])
    # X= (X-np.mean(X,axis=(1,2)).reshape(-1,1,1))/np.std(X,axis=(1,2)).reshape(-1,1,1)
    X = X.reshape(X.shape+(1,))
    if y is not None:
        # Convert to one-hot encoding. Convert back with `y = y.nonzero()[1]`
        y = np.eye(43)[y]
        # Shuffle the data
        X, y = shuffle(X, y)
    # Add a single grayscale channel
    return X, y


# parameters


def get_time_hhmmss(start=None):
    """
    Calculates time since `start` and formats as a string.
    """
    if start is None:
        return time.strftime("%Y/%m/%d %H:%M:%S")
    end = time.time()
    m, s = divmod(end - start, 60)
    h, m = divmod(m, 60)
    time_str = "%02d:%02d:%02d" % (h, m, s)
    return time_str


def load_pickled_data(file, columns):
    """
    Loads pickled training and test data.

    Parameters
    ----------
    file    :
              Name of the pickle file.
    columns : list of strings
              List of columns in pickled data we're interested in.

    Returns
    -------
    A tuple of datasets for given columns.
    """

    with open(file, mode='rb') as f:
        dataset = pickle.load(f)
    return tuple(map(lambda c: dataset[c], columns))


def load_and_preprocess_data():
    """
    Loads pickled data and preprocesses images and labels by scaling features,
    shuffling the data and applying one-hot encoding to labels.

    Parameters
    ----------

    Returns
    -------
    X_train, y_train, X_valid, y_valid, X_test, y_test
    """
    training_file = "traffic-signs-data/train.p"
    validation_file = "traffic-signs-data/valid.p"
    testing_file = "traffic-signs-data/test.p"

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'].astype(float), train['labels']
    X_valid, y_valid = valid['features'].astype(float), valid['labels']
    X_test, y_test = test['features'].astype(float), test['labels']

    X_train, y_train = preprocess_dataset(X_train, y_train)
    X_valid, y_valid = preprocess_dataset(X_valid, y_valid)
    X_test, y_test = preprocess_dataset(X_test, y_test)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


Parameters = namedtuple('Parameters', [
    # Data parameters
    'num_classes', 'image_size',
    # Training parameters
    'batch_size', 'max_epochs', 'log_epoch', 'print_epoch',
    # Optimisations
    'learning_rate_decay', 'learning_rate',
    'l2_reg_enabled', 'l2_lambda',
    'early_stopping_enabled', 'early_stopping_patience',
    'resume_training',
    # Layers architecture
    'conv1_k', 'conv1_d', 'conv1_p',
    'conv2_k', 'conv2_d', 'conv2_p',
    'conv3_k', 'conv3_d', 'conv3_p',
    'conv4_k', 'conv4_d', 'conv4_p',
    'fc5_size', 'fc5_p',
    'out_p'
])


class Paths(object):
    """
    Provides easy access to common paths we use for persisting
    the data associated with model training.
    """

    def __init__(self, params):
        """
        Initialises a new `Paths` instance and creates corresponding folders if needed.

        Parameters
        ----------
        params  : Parameters
                  Structure (`namedtuple`) containing model parameters.
        """
        self.model_name = self.get_model_name(params)
        self.var_scope = self.get_variables_scope(params)
        self.root_path = os.getcwd() + "/models/" + self.model_name + "/"
        self.model_path = self.get_model_path()
        self.train_history_path = self.get_train_history_path()
        self.learning_curves_path = self.get_learning_curves_path()
        os.makedirs(self.root_path, exist_ok=True)

    def get_model_name(self, params):
        """
        Generates a model name with some of the crucial model parameters encoded into the name.

        Parameters
        ----------
        params  : Parameters
                  Structure (`namedtuple`) containing model parameters.

        Returns
        -------
        Model name.
        """
        # We will encode model settings in its name: architecture, optimisations applied, etc.
        model_name = "k{}d{}p{}_k{}d{}p{}_k{}d{}p{}_k{}d{}p{}_fc{}p{}_out_p{}".format(
            params.conv1_k, params.conv1_d, params.conv1_p,
            params.conv2_k, params.conv2_d, params.conv2_p,
            params.conv3_k, params.conv3_d, params.conv3_p,
            params.conv4_k, params.conv4_d, params.conv4_p, 
            params.fc5_size, params.fc5_p,
            params.out_p
        )
        model_name += "_lrdec" if params.learning_rate_decay else "_no-lrdec"
        model_name += "_l2" if params.l2_reg_enabled else "_no-l2"
        return model_name

    def get_variables_scope(self, params):
        """
        Generates a model variable scope with some of the crucial model parameters encoded.

        Parameters
        ----------
        params  : Parameters
                  Structure (`namedtuple`) containing model parameters.

        Returns
        -------
        Variables scope name.
        """
        # We will encode model settings in its name: architecture, optimisations applied, etc.
        var_scope = "k{}d{}_k{}d{}_k{}d{}_fc{}_fc0".format(
            params.conv1_k, params.conv1_d,
            params.conv2_k, params.conv2_d,
            params.conv3_k, params.conv3_d,
            params.conv4_k, params.conv4_d,
            params.fc5_size
        )
        return var_scope

    def get_model_path(self):
        """
        Generates path to the model file.

        Returns
        -------
        Model file path.
        """
        return self.root_path + "model.ckpt"

    def get_train_history_path(self):
        """
        Generates path to the train history file.

        Returns
        -------
        Train history file path.
        """
        return self.root_path + "train_history"

    def get_learning_curves_path(self):
        """
        Generates path to the learning curves graph file.

        Returns
        -------
        Learning curves file path.
        """
        return self.root_path + "learning_curves.png"


class EarlyStopping(object):
    """
    Provides early stopping functionality. Keeps track of model accuracy,
    and if it doesn't improve over time restores last best performing
    parameters.
    """

    def __init__(self, saver, session, patience=100, minimize=True):
        """
        Initialises a `EarlyStopping` isntance.

        Parameters
        ----------
        saver     :
                    TensorFlow Saver object to be used for saving and restoring model.
        session   :
                    TensorFlow Session object containing graph where model is restored.
        patience  :
                    Early stopping patience. This is the number of epochs we wait for
                    accuracy to start improving again before stopping and restoring
                    previous best performing parameters.

        Returns
        -------
        New instance.
        """
        self.minimize = minimize
        self.patience = patience
        self.saver = saver
        self.session = session
        self.best_monitored_value = np.inf if minimize else 0.
        self.best_monitored_epoch = 0
        self.restore_path = None

    def __call__(self, value, epoch):
        """
        Checks if we need to stop and restores the last well performing values if we do.

        Parameters
        ----------
        value     :
                    Last epoch monitored value.
        epoch     :
                    Last epoch number.

        Returns
        -------
        `True` if we waited enough and it's time to stop and we restored the
        best performing weights, or `False` otherwise.
        """
        if (self.minimize and value < self.best_monitored_value) or (
                    not self.minimize and value > self.best_monitored_value):
            self.best_monitored_value = value
            self.best_monitored_epoch = epoch
            self.restore_path = self.saver.save(self.session,
                                                os.getcwd() + "/early_stopping_checkpoint")
        elif self.best_monitored_epoch + self.patience < epoch:
            if self.restore_path != None:
                self.saver.restore(self.session, self.restore_path)
            else:
                print("ERROR: Failed to restore session")
            return True

        return False


class ModelCloudLog(CloudLog):
    def log_parameters(self, params, train_size, valid_size, test_size):
        """
        Logs model parameters to console and appends the same text representation to the log file.

        Parameters
        ----------
        params    : Parameters
                    Structure (`namedtuple`) containing model parameters.
        train_size: int
                    Size of the training dataset.
        valid_size: int
                    Size of the training dataset.
        test_size : int
                    Size of the training dataset.
        """
        if params.resume_training:
            self("=============================================")
            self("============= RESUMING TRAINING =============")
            self("=============================================")

        self("=================== DATA ====================")
        self("            Training set: {} examples".format(train_size))
        self("          Validation set: {} examples".format(valid_size))
        self("             Testing set: {} examples".format(test_size))
        self("              Batch size: {}".format(params.batch_size))

        self("=================== MODEL ===================")
        self("--------------- ARCHITECTURE ----------------")
        self(" %-*s %-*s %-*s %-*s" % (10, "", 10, "Type", 8, "Size", 15, "Dropout (keep p)"))
        self(" %-*s %-*s %-*s %-*s" % (
            10, "Layer 1", 10, "{}x{} Conv".format(params.conv1_k, params.conv1_k), 8,
            str(params.conv1_d), 15, str(params.conv1_p)))
        self(" %-*s %-*s %-*s %-*s" % (
            10, "Layer 2", 10, "{}x{} Conv".format(params.conv2_k, params.conv2_k), 8,
            str(params.conv2_d), 15, str(params.conv2_p)))
        self(" %-*s %-*s %-*s %-*s" % (
            10, "Layer 3", 10, "{}x{} Conv".format(params.conv3_k, params.conv3_k), 8,
            str(params.conv3_d), 15, str(params.conv3_p)))
        self(" %-*s %-*s %-*s %-*s" % (
            10, "Layer 4", 10, "{}x{} Conv".format(params.conv4_k, params.conv4_k), 8,
            str(params.conv4_d), 15, str(params.conv4_p)))
        self(" %-*s %-*s %-*s %-*s" % (
            10, "Layer 5", 10, "FC", 8, str(params.fc5_size), 15, str(params.fc5_p)))
        self(" %-*s %-*s %-*s %-*s" % (
            10, "Output", 10, "FC", 8, str(params.num_classes), 15, str(params.out_p)))
        self("---------------- PARAMETERS -----------------")
        self("     Learning rate decay: " + (
            "Enabled" if params.learning_rate_decay else "Disabled (rate = {})".format(
                params.learning_rate)))
        self("       L2 Regularization: " + (
            "Enabled (lambda = {})".format(
                params.l2_lambda) if params.l2_reg_enabled else "Disabled"))
        self("          Early stopping: " + ("Enabled (patience = {})".format(
            params.early_stopping_patience) if params.early_stopping_enabled else "Disabled"))
        self(" Keep training old model: " + ("Enabled" if params.resume_training else "Disabled"))


def fully_connected(input, size):
    """
    Performs a single fully connected layer pass, e.g. returns `input * weights + bias`.
    """
    weights = tf.get_variable('weights',
                              shape=[input.get_shape()[1], size],
                              initializer=tf.contrib.layers.xavier_initializer()
                              )
    biases = tf.get_variable('biases',
                             shape=[size],
                             initializer=tf.constant_initializer(0.1)
                             )
    return tf.matmul(input, weights) + biases


def fully_connected_relu(input, size):
    return tf.nn.relu(fully_connected(input, size))


def conv_relu(input, kernel_size, depth):
    """
    Performs a single convolution layer pass.
    """
    weights = tf.get_variable('weights',
                              shape=[kernel_size, kernel_size, input.get_shape()[3], depth],
                              initializer=tf.contrib.layers.xavier_initializer()
                              )
    biases = tf.get_variable('biases',
                             shape=[depth],
                             initializer=tf.constant_initializer(0.1)
                             )
    conv = tf.nn.conv2d(input, weights,
                        strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + biases)


def pool(input, size):
    """
    Performs a max pooling layer pass.
    """
    return tf.nn.max_pool(
        input,
        ksize=[1, size, size, 1],
        strides=[1, size, size, 1],
        padding='SAME'
    )


def model_pass(input, params, is_training):
    """
    Performs a full model pass.

    Parameters
    ----------
    input         : Tensor
                    NumPy array containing a batch of examples.
    params        : Parameters
                    Structure (`namedtuple`) containing model parameters.
    is_training   : Tensor of type tf.bool
                    Flag indicating if we are training or not (e.g. whether to use dropout).

    Returns
    -------
    Tensor with predicted logits.
    """
    # Convolutions

    with tf.variable_scope('conv1'):
        conv1 = conv_relu(input, kernel_size=params.conv1_k, depth=params.conv1_d)
    with tf.variable_scope('pool1'):
        pool1 = pool(conv1, size=2)
        pool1 = tf.cond(is_training, lambda: tf.nn.dropout(pool1, keep_prob=params.conv1_p),
                        lambda: pool1)
    with tf.variable_scope('conv2'):
        conv2 = conv_relu(pool1, kernel_size=params.conv2_k, depth=params.conv2_d)
    with tf.variable_scope('pool2'):
        pool2 = pool(conv2, size=2)
        pool2 = tf.cond(is_training, lambda: tf.nn.dropout(pool2, keep_prob=params.conv2_p),
                        lambda: pool2)
    with tf.variable_scope('conv3'):
        conv3 = conv_relu(pool2, kernel_size=params.conv3_k, depth=params.conv3_d)
    with tf.variable_scope('pool3'):
        pool3 = pool(conv3, size=2)
        pool3 = tf.cond(is_training, lambda: tf.nn.dropout(pool3, keep_prob=params.conv3_p),
                        lambda: pool3)
    with tf.variable_scope('conv4'):
        conv4 = conv_relu(pool3, kernel_size=params.conv4_k, depth=params.conv4_d)
    with tf.variable_scope('pool4'):
        pool4 = pool(conv4, size=2)
        pool4 = tf.cond(is_training, lambda: tf.nn.dropout(pool4, keep_prob=params.conv4_p),
                        lambda: pool4)

    # Fully connected

    # 1st stage output
    # pool1 = pool(pool1, size=4)
    # shape = pool1.get_shape().as_list()
    # pool1 = tf.reshape(pool1, [-1, shape[1] * shape[2] * shape[3]])

    # # 2nd stage output
    pool2 = pool(pool2, size=4)
    shape = pool2.get_shape().as_list()
    pool2 = tf.reshape(pool2, [-1, shape[1] * shape[2] * shape[3]])

    # pool3 = pool(pool3, size=2)
    # shape = pool3.get_shape().as_list()
    # pool3 = tf.reshape(pool3, [-1, shape[1] * shape[2] * shape[3]])

    shape = pool4.get_shape().as_list()
    pool4 = tf.reshape(pool4, [-1, shape[1] * shape[2] * shape[3]])

    flattened = tf.concat([pool2, pool4], 1)

    # shape = input.get_shape().as_list()
    # pool3 = tf.reshape(input, [-1, shape[1] * shape[2] * shape[3]])

    with tf.variable_scope('fc5'):
        fc5 = fully_connected_relu(flattened, size=params.fc5_size)
        fc5 = tf.cond(is_training, lambda: tf.nn.dropout(fc5, keep_prob=params.fc5_p), lambda: fc5)
    with tf.variable_scope('out'):
        logits = fully_connected(fc5, size=params.num_classes)
        logits = tf.cond(is_training, lambda: tf.nn.dropout(logits, keep_prob=params.out_p), lambda: logits)
    return logits


def plot_curve(axis, params, train_column, valid_column, linewidth=2, train_linestyle="b-",
               valid_linestyle="g-"):
    """
    Plots a pair of validation and training curves on a single plot.
    """
    model_history = np.load(Paths(params).train_history_path + ".npz")
    train_values = model_history[train_column]
    valid_values = model_history[valid_column]
    epochs = train_values.shape[0]
    x_axis = np.arange(epochs)
    axis.plot(x_axis[train_values > 0], train_values[train_values > 0], train_linestyle,
              linewidth=linewidth, label="train")
    axis.plot(x_axis[valid_values > 0], valid_values[valid_values > 0], valid_linestyle,
              linewidth=linewidth, label="valid")
    return epochs


# Plots history of learning curves for a specific model.
def plot_learning_curves(parameters):
    """
    Plots learning curves (loss and accuracy on both training and validation sets) for a model
    identified by a parameters struct.
    """
    curves_figure = pyplot.figure(figsize=(10, 4))
    axis = curves_figure.add_subplot(1, 2, 1)
    epochs_plotted = plot_curve(axis, parameters, train_column="train_accuracy_history",
                                valid_column="valid_accuracy_history")

    pyplot.grid()
    pyplot.legend()
    pyplot.xlabel("epoch")
    pyplot.ylabel("accuracy")
    pyplot.ylim(50., 115.)
    pyplot.xlim(0, epochs_plotted)

    axis = curves_figure.add_subplot(1, 2, 2)
    epochs_plotted = plot_curve(axis, parameters, train_column="train_loss_history",
                                valid_column="valid_loss_history")

    pyplot.grid()
    pyplot.legend()
    pyplot.xlabel("epoch")
    pyplot.ylabel("loss")
    pyplot.ylim(0.0001, 10.)
    pyplot.xlim(0, epochs_plotted)
    pyplot.yscale("log")


def train_model(params, X_train, y_train, X_valid, y_valid, X_test, y_test):
    """
    Performs model training based on provided training dataset
    according to provided parameters, and then evaluates trained
    model with testing dataset.
    Part of the training dataset may be used for validation during
    training if specified in model parameters.

    Parameters
    ----------
    params        : Parameters
                    Structure (`namedtuple`) containing model parameters.
    X_train       :
                    Training dataset.
    y_train       :
                    Training dataset labels.
    X_valid       :
                    Validation dataset.
    y_valid       :
                    Validation dataset labels.
    X_test        :
                    Testing dataset.
    y_test        :
                    Testing dataset labels.
    logger_config :
                    Logger configuration, containing Dropbox and Telegram settings
                    for notifications and cloud logs backup.
    """

    # Initialisation routines: generate variable scope, create logger, note start time.
    paths = Paths(params)
    log = ModelCloudLog(
        os.path.join(paths.root_path, "logs"),
    )
    start = time.time()
    model_variable_scope = paths.var_scope

    log.log_parameters(params, y_train.shape[0], y_valid.shape[0], y_test.shape[0])

    # image augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        shear_range=0.5,
        fill_mode='nearest')

    # Build the graph
    graph = tf.Graph()
    with graph.as_default():
        # Input data. For the training data, we use a placeholder that will be fed at run time
        # with a training minibatch.
        tf_x_batch = tf.placeholder(tf.float32,
                                    shape=(None, params.image_size[0], params.image_size[1], 1))
        tf_y_batch = tf.placeholder(tf.float32, shape=(None, params.num_classes))
        is_training = tf.placeholder(tf.bool)
        current_epoch = tf.Variable(0, trainable=False)  # count the number of epochs

        # Model parameters.
        if params.learning_rate_decay:
            learning_rate = tf.train.exponential_decay(params.learning_rate, current_epoch,
                                                       decay_steps=params.max_epochs,
                                                       decay_rate=0.01)
        else:
            learning_rate = params.learning_rate

        # Training computation.
        with tf.variable_scope(model_variable_scope):
            logits = model_pass(tf_x_batch, params, is_training)
            if params.l2_reg_enabled:
                with tf.variable_scope('fc5', reuse=True):
                    l2_loss = tf.nn.l2_loss(tf.get_variable('weights'))
            else:
                l2_loss = 0

        predictions = tf.nn.softmax(logits)
        softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                        labels=tf_y_batch)
        loss = tf.reduce_mean(softmax_cross_entropy) + params.l2_lambda * l2_loss

        # Optimizer.
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate
        ).minimize(loss)

    with tf.Session(graph=graph) as session:
        session.run(tf.global_variables_initializer())

        # A routine for evaluating current model parameters
        def get_accuracy_and_loss_in_batches(X, y):
            p = []
            sce = []
            batch_iterator = BatchIterator(batch_size=256)
            for x_batch, y_batch in batch_iterator(X, y):
                [p_batch, sce_batch] = session.run([predictions, softmax_cross_entropy], feed_dict={
                    tf_x_batch: x_batch,
                    tf_y_batch: y_batch,
                    is_training: False
                }
                                                   )
                p.extend(p_batch)
                sce.extend(sce_batch)
            p = np.array(p)
            sce = np.array(sce)
            accuracy = 100.0 * np.sum(np.argmax(p, 1) == np.argmax(y, 1)) / p.shape[0]
            loss = np.mean(sce)
            return accuracy, loss

        # If we chose to keep training previously trained model, restore session.
        if params.resume_training:
            try:
                tf.train.Saver().restore(session, paths.model_path)
            except Exception as e:
                log("Failed restoring previously trained model: file does not exist.")
                pass

        saver = tf.train.Saver()
        early_stopping = EarlyStopping(tf.train.Saver(), session,
                                       patience=params.early_stopping_patience, minimize=True)
        train_loss_history = np.empty([0], dtype=np.float32)
        train_accuracy_history = np.empty([0], dtype=np.float32)
        valid_loss_history = np.empty([0], dtype=np.float32)
        valid_accuracy_history = np.empty([0], dtype=np.float32)
        if params.max_epochs > 0:
            log("================= TRAINING ==================")
        else:
            log("================== TESTING ==================")
        log(" Timestamp: " + get_time_hhmmss())
        log.sync()

        for epoch in range(params.max_epochs):
            current_epoch = epoch
            batch_ct=0
            # Train on whole randomised dataset in batches
            # batch_iterator = BatchIterator(batch_size=params.batch_size, shuffle=True)
            for x_batch, y_batch in datagen.flow(X_train, y_train, batch_size=params.batch_size):
                batch_ct += 1
                session.run([optimizer], feed_dict={
                    tf_x_batch: x_batch,
                    tf_y_batch: y_batch,
                    is_training: True
                })
                if batch_ct> (X_train.shape[0] / params.batch_size):
                    break

            # If another significant epoch ended, we log our losses.
            if (epoch % params.log_epoch == 0):
                # Get validation data predictions and log validation loss:
                valid_accuracy, valid_loss = get_accuracy_and_loss_in_batches(X_valid, y_valid)

                # Get training data predictions and log training loss:
                train_accuracy, train_loss = get_accuracy_and_loss_in_batches(X_train, y_train)

                if (epoch % params.print_epoch == 0):
                    log("-------------- EPOCH %4d/%d --------------" % (epoch, params.max_epochs))
                    log("     Train loss: %.8f, accuracy: %.2f%%" % (train_loss, train_accuracy))
                    log("Validation loss: %.8f, accuracy: %.2f%%" % (valid_loss, valid_accuracy))
                    log("      Best loss: %.8f at epoch %d" % (
                        early_stopping.best_monitored_value, early_stopping.best_monitored_epoch))
                    log("   Elapsed time: " + get_time_hhmmss(start))
                    log("      Timestamp: " + get_time_hhmmss())
                    log.sync()
            else:
                valid_loss = 0.
                valid_accuracy = 0.
                train_loss = 0.
                train_accuracy = 0.

            valid_loss_history = np.append(valid_loss_history, [valid_loss])
            valid_accuracy_history = np.append(valid_accuracy_history, [valid_accuracy])
            train_loss_history = np.append(train_loss_history, [train_loss])
            train_accuracy_history = np.append(train_accuracy_history, [train_accuracy])

            if params.early_stopping_enabled:
                # Get validation data predictions and log validation loss:
                if valid_loss == 0:
                    _, valid_loss = get_accuracy_and_loss_in_batches(X_valid, y_valid)
                if early_stopping(valid_loss, epoch):
                    log("Early stopping.\nBest monitored loss was {:.8f} at epoch {}.".format(
                        early_stopping.best_monitored_value, early_stopping.best_monitored_epoch
                    ))
                    break

        # Evaluate on test dataset.
        test_accuracy, test_loss = get_accuracy_and_loss_in_batches(X_test, y_test)
        valid_accuracy, valid_loss = get_accuracy_and_loss_in_batches(X_valid, y_valid)
        log("=============================================")
        log(" Valid loss: %.8f, accuracy = %.2f%%)" % (valid_loss, valid_accuracy))
        log(" Test loss: %.8f, accuracy = %.2f%%)" % (test_loss, test_accuracy))
        log(" Total time: " + get_time_hhmmss(start))
        log("  Timestamp: " + get_time_hhmmss())

        # Save model weights for future use.
        saved_model_path = saver.save(session, paths.model_path)
        log("Model file: " + saved_model_path)
        np.savez(paths.train_history_path, train_loss_history=train_loss_history,
                 train_accuracy_history=train_accuracy_history,
                 valid_loss_history=valid_loss_history,
                 valid_accuracy_history=valid_accuracy_history)
        log("Train history file: " + paths.train_history_path)
        log.sync(notify=True,
                 message="Finished training with *%.2f%%* accuracy on the testing set (loss = "
                         "*%.6f*)." % (
                             test_accuracy, test_loss))

        plot_learning_curves(params)
        log.add_plot(notify=True, caption="Learning curves")

        pyplot.show()


def predict_prob(params, X):
    """
    Evaluates `X` on a model defined by `params` and returns top 5 predictions.

    Parameters
    ----------
    params    : Parameters
                Structure (`namedtuple`) containing model parameters.
    X         :
                Testing dataset.
    k         :
                Number of top predictions we are interested in.

    Returns
    -------
    An array of top k softmax predictions for each example.
    """

    # Initialisation routines: generate variable scope, create logger, note start time.
    paths = Paths(params)

    # Build the graph
    graph = tf.Graph()
    with graph.as_default():
        # Input data. For the training data, we use a placeholder that will be fed at run time
        # with a training minibatch.
        tf_x_batch = tf.placeholder(tf.float32,
                                    shape=(None, params.image_size[0], params.image_size[1], 1))
        is_training = tf.constant(False)
        with tf.variable_scope(paths.var_scope):
            predictions = tf.nn.sigmoid(model_pass(tf_x_batch, params, is_training))
            # top_k_predictions = tf.nn.top_k(predictions, k)
    with tf.Session(graph=graph) as session:
        # session.run(tf.global_variables_initializer())
        tf.train.Saver().restore(session, paths.model_path)
        p = []
        batch_iterator = BatchIterator(batch_size=256)
        for x_batch, y_batch in batch_iterator(X):
            [p_batch] = session.run([predictions], feed_dict={
                tf_x_batch: x_batch,
                is_training: False
            }
                                    )
            p.extend(p_batch)
        p = np.array(p)

        return p
        