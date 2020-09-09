import numpy as np
import sys

sys.path.append('mytorch')
from gru_cell import *
from linear import *

# This is the neural net that will run one timestep of the input
# You only need to implement the forward method of this class.
# This is to test that your GRU Cell implementation is correct when used as a GRU.
class CharacterPredictor(object):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(CharacterPredictor, self).__init__()
        # The network consists of a GRU Cell and a linear layer
        self.rnn = GRU_Cell(input_dim, hidden_dim)
        self.projection = Linear(hidden_dim, num_classes)
        self.num_classes = num_classes

    def init_rnn_weights(self, w_hi, w_hr, w_hn, w_ii, w_ir, w_in):
        # DO NOT MODIFY
        self.rnn.init_weights(w_hi, w_hr, w_hn, w_ii, w_ir, w_in)

    def __call__(self, x, h):
        return self.forward(x, h)

    def forward(self, x, h):
        # A pass through one time step of the input
        h_t = self.rnn(x,h)
        output = self.projection(h_t)
        return  output, h_t
# An instance of the class defined above runs through a sequence of inputs to
# generate the logits for all the timesteps.
def inference(net, inputs):
    # input:
    #  - net: An instance of CharacterPredictor
    #  - inputs - a sequence of inputs of dimensions [seq_len x feature_dim]
    # output:
    #  - logits - one per time step of input. Dimensions [seq_len x num_classes]
    #logits = np.zeros((inputs.shape[0],self.num_classes))
    logits = []
    for t in range(inputs.shape[0]):
        if t == 0:
            h_t = np.zeros(net.rnn.h)
        logit, h_t = net(inputs[t],h_t)
        logits.append(logit)
    return np.asarray(logits)
