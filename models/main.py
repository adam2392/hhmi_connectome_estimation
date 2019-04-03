import numpy as np
import keras
import tensorflow as tf

class Encoder(tf.keras.Model):
    """
    Class for the encoder - artificial neural network based

    This module attempts to approximate a possibly nonlinear function, f,
    which maps: x -> Y (MxM matrix, where M is number_neurons).

    Attributes
    ----------
    number_neurons : int
        The number of neurons in your corresponding neuronal network.
    enc_units : int
        The number of units in your encoder
    num_layers : int
        The number of layers, default as 1
    batch_size : int
        Whether to perform estimation of the A matrix using a regularization term.
    Notes
    -----

    When the size of the data is too large (e.g. N > 180, W > 1000), then right now the construction of the csr
    matrix scales up. With more efficient indexing, we can perhaps decrease this.

    Examples
    --------
    >>> import numpy as np
    >>> from models.main import Encoder
    >>> model_params = {
    ...     'number_neurons': 5,
    ...     'num_layers': 5,
    ...     'enc_units': 50,
    ...     'batch_size': 8,
    ...     }
    >>> model = Encoder(**model_params)
    """
    def __init__(self, number_neurons, num_layers=1, enc_units=124, batch_size=8):
        super(Encoder, self).__init__()

        self.number_neurons = number_neurons
        self.num_layers = num_layers

        self.batch_size = batch_size
        self.enc_units = enc_units

        # define the model
        self.layers = [tf.keras.layers.LSTM(self.enc_units,
                                          return_sequences=True,
                                          return_state=True,
                                          recurrent_initializer='glorot_uniform')]*num_layers

    def call(self, x, hidden):
        output, state = self.model(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.enc_units))

