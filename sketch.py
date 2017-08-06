import theano
import theano.tensor as T

import numpy as np 

from six.moves import cPickle

# sketch-rnn implementation

class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size 
        self.hidden_size = hidden_size 

        self.weights = {
                            "f:x" : init_weight(self.input_size, self.hidden_size),
                            "f:h" : init_weight(self.hidden_size, self.hidden_size),
                            "f:b" : init_weight(self.hidden_size),
                            "i:x" : init_weight(self.input_size, self.hidden_size),
                            "i:h" : init_weight(self.hidden_size, self.hidden_size),
                            "i:b" : init_weight(self.hidden_size),
                            "o:x" : init_weight(self.input_size, self.hidden_size),
                            "o:h" : init_weight(self.hidden_size, self.hidden_size),
                            "o:b" : init_weight(self.hidden_size),
                            "c:x" : init_weight(self.input_size, self.hidden_size),
                            "c:h" : init_weight(self.hidden_size, self.hidden_size),
                            "c:b" : init_weight(self.hidden_size),
                       }
    
    def get_weights(self):
        return [self.weights[key] for key in self.weights.keys()]

    def recurrence(self, inp, prev_hidden, prev_cell):
        """
            LSTM.recurrence(input_, prev_hidden, prev_cell) -> hidden, cell (batchsize x hidden_size)

            Produces the new hidden and cell state, acting as a single computation step of an LSTM

            @param input_: a batchsize x input_size matrix that represents the new data to input into the LSTM
            @param prev_hidden: a batchsize x hidden_size matrix that represents the previous hidden state of the network
            @param prev_cell: a batchsize x hidden_size matrix that represents the previous cell state of the network
        """

        forget = T.nnet.sigmoid(T.dot(inp, self.weights["f:x"]) +\
                                T.dot(prev_hidden, self.weights["f:h"]) +\
                                self.weights["f:b"])
        
        input_ = T.nnet.sigmoid(T.dot(inp, self.weights["i:x"]) +\
                                T.dot(prev_hidden, self.weights["i:h"]) +\
                                self.weights["i:b"])
        
        output = T.nnet.sigmoid(T.dot(inp, self.weights["o:x"]) +\
                                T.dot(prev_hidden, self.weights["o:h"]) +\
                                self.weights["o:b"])
        
        cell = T.mul(forget, prev_cell) + T.mul(input_, T.tanh(T.dot(inp, self.weights["c:x"]) +\
                                                                     T.dot(prev_hidden, self.weights["c:h"]) +\
                                                                     self.weights["c:b"]))
        
        hidden = T.mul(output, cell)

        return hidden, cell 

def pdf(x, y, x_mean, y_mean, x_variance, y_variance, correlation):
    z = ((x - x_mean) ** 2)/x_variance ** 2 - (2 * correlation * (x - x_mean) * (y - y_mean))/(x_variance * y_variance) + ((y - y_mean) ** 2)/(y_variance ** 2)

    p = 1/(2 * 3.1415926 * x_variance * y_variance * T.sqrt(1 - correlation ** 2)) * T.exp(-z/(2 * (1 - correlation ** 2)))

    return p

class decoderLSTM:
    def __init__(self, input_size, hidden_size):
        self.decLSTM = LSTM(input_size + 101, hidden_size)

        # 10 mixtures, 6 parameters each, and 3 for the p1, p2 and p3 states 
        self.hidden_GMM_params = init_weight(hidden_size, 6 * 10 + 3)
    
    def get_weights(self):
        return self.decLSTM.get_weights() + [self.hidden_GMM_params]

    def process_step(self, data, hidden, cell, latent):
        """
            @param gaussian_x: batchsize x 10 matrix
        """
        concatenated_state = T.concatenate([data, latent], axis = 1)
        hidden, cell = self.decLSTM.recurrence(concatenated_state, hidden, cell)

        mixture_params = T.dot(hidden, self.hidden_GMM_params)

        # first ten will be the weightings
        weightings = T.nnet.softmax(mixture_params[:, :10])
        x_mean = mixture_params[:, 10:10 + 10]
        y_mean = mixture_params[:, 20 : 20 + 10]
        x_variance = T.exp(mixture_params[:, 30 : 30 + 10])
        y_variance = T.exp(mixture_params[:, 40 : 40 + 10])

        correlation = T.tanh(mixture_params[:, 50 : 50 + 10])

        p_values = T.nnet.softmax(mixture_params[:, 60 : 63])

        return hidden, cell, x_mean, y_mean, x_variance, y_variance, correlation, p_values, weightings

    # sequences prior results non sequences
    def process(self, data, latent_vector, initial_hidden, initial_cell):
        [hidden_states, _, x_mean, y_mean, x_variance, y_variance, correlation, p_values, weightings], updates = theano.scan(fn = self.process_step, sequences = [data], outputs_info = [initial_hidden, initial_cell, None, None, None, None, None, None, None], non_sequences = [latent_vector])

        return x_mean, y_mean, x_variance, y_variance, correlation, p_values, weightings

class bidirectionalLSTM:
    def __init__(self, input_size, hidden_size, batch_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.forward = LSTM(input_size, hidden_size)
        self.backward = LSTM(input_size, hidden_size)

        self.batch_size = batch_size

    def get_weights(self):
        return self.forward.get_weights() + self.backward.get_weights()

    def process_step(self, forward_data, backward_data, forward_hidden, forward_cell, backward_hidden, backward_cell):
        forward_hidden, forward_cell = self.forward.recurrence(forward_data, forward_hidden, forward_cell)
        backward_hidden, backward_cell = self.backward.recurrence(backward_data, backward_hidden, backward_cell)

        return forward_hidden, forward_cell, backward_hidden, backward_cell 

    def process(self, data):
        """
            bidirectionalLSTM.process(data) -> forward_states   (timesteps x batchsize x hidden_size)
                                               backward_states  (timsteps x batchsize x hidden_size)
            
            processes data

            @param data: a timestep x batchsize x input_size 3-tensor that represents the data we're inputting.
        """

        # Flip data in the timestep axis
        [forward_states, _, backward_states, _], updates = theano.scan(fn = self.process_step, sequences = [data, data[::-1]], outputs_info = [T.zeros([self.batch_size, self.hidden_size]), T.zeros([self.batch_size, self.hidden_size]),T.zeros([self.batch_size, self.hidden_size]), T.zeros([self.batch_size, self.hidden_size])])

        return forward_states, backward_states

def init_weight(input_size, output_size = None):
    if output_size == None:
        return theano.shared(np.zeros(shape = [input_size]))
    else:
        return theano.shared((np.random.randn(input_size, output_size) * np.sqrt(3 / (input_size + output_size))))

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / (gradient_scaling + 1e-10)
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def main():
    latent_mean_fc = init_weight(202, 101)
    latent_log_variance_fc = init_weight(202, 101)
    latent_initial_fc = init_weight(101, 202)

    # timestep x batchsize x 5
    # we can ensure every sketch has the same amount of timesteps by padding them with [0, 0, 0, 0, 1]s
    sketch = T.tensor3()

    # timestep x batchsize x 5
    # same thing as sketch, but shifted backwards one timestep. That way, our target is just `sketch`
    # the first timestep is just (0, 0, 1, 0, 0)
    dec_sketch = T.tensor3()

    # batchsize x 100
    gaussian_sample = T.matrix()
    
    biLSTM = bidirectionalLSTM(5, 101, 2)

    forward_states, backward_states = biLSTM.process(sketch)

    # timestep x batchsize x 100 each     
    final_forward_state = forward_states[-1]
    final_backward_state = backward_states[-1]

    # batchsize x 200
    concatenated_state = T.concatenate([final_forward_state, final_backward_state], axis = 1)

    latent_means = T.dot(concatenated_state, latent_mean_fc)
    latent_log_variances = T.dot(concatenated_state, latent_log_variance_fc)
    latent_variances = T.exp(latent_log_variances / 2)

    latent_vector = gaussian_sample * latent_variances + latent_means 

    initial_hidden = T.tanh(T.dot(latent_vector, latent_initial_fc))
    initial_cell = T.tanh(T.dot(latent_vector, latent_initial_fc))

    decLSTM = decoderLSTM(5, 202)

    x_mean, y_mean, x_variance, y_variance, correlation, p_values, weightings = decLSTM.process(data = dec_sketch, latent_vector = latent_vector, initial_hidden = initial_hidden, initial_cell = initial_cell)

    # Find the probability of the correct states for each sketch value, with its corresponding x,y mean and variance + correlation. Then multiply by weighting and find log.

    # Each of the terms x_mean, y_mean, x_variance, y_variance and correlation will have shapes timestep x batchsize x 10. x and y will have shapes timestep x batchsize x 1 each. So x and y will have to be broadcasted on the last axis.

    sketch_x = T.addbroadcast(sketch[:, :, 0:1], 2)
    sketch_y = T.addbroadcast(sketch[:, :, 1:2], 2)

    # Output should be a timestep x batchsize x 10 3-tensor
    pdf_loss = pdf(sketch_x, sketch_y, x_mean, y_mean, x_variance, y_variance, correlation)

    # weightings is a timestep x batchsize x 10 3-tensor 
    # Once we've weighted the pdf_loss, we need to add along the last axis to get the weighted average
    weighted_pdf_loss = T.log(T.sum(weightings * pdf_loss, axis = 2) + 1e-8)

    # Sum along timesteps, we're basically taking an average, but we're normalizing by a constant that will often be larger than the number of timesteps (`MAX_TIMESTEPS`)
    # resulting shape is batchsize 
    
    weighted_pdf_loss = T.sum(weighted_pdf_loss, axis = 0)/300

    # average over the losses, to yield the offset loss--the loss incurred by the network predicting the offsets, or, the values of `sketch`
    offset_loss = -T.mean(weighted_pdf_loss)

    # Each of the p_values model a bernoulli distribution. We are trying to minimize the negative log loss of the predicted p_values and the real p_values:
    # log(P(p_real)) -> log(P(p_real_1) * P(p_real_2) * p(p_real_3) -> sum(log(P(p_real_x)) x: 1->3

    # timestep x batchsize x 3
    real_p_values = sketch[:, :, 2:]

    # p_values is the probability p_x will be 1
    # 1 - p_values is the probability p_x will be 0

    # Summing along the last axis reduces log_loss to timestep x batchsize
    log_loss = T.sum(T.log(p_values)  * real_p_values , axis = 2) + T.sum(T.log((1 - p_values)) * (1 - real_p_values), axis = 2)   

    # Sum along timestep axis, and then take the mean
    log_loss = -T.mean(T.sum(log_loss, axis = 0))

    reconstruction_loss = log_loss + offset_loss

    # latent_means and latent_variances are batchsize x 100 
    latent_loss = T.mean(T.sum(-1/200 * (latent_means ** 2 + latent_variances ** 2 - T.log(latent_variances ** 2)) - .5, axis = 1), axis = 0)

    total_loss = reconstruction_loss + latent_loss

    params = [latent_mean_fc, latent_log_variance_fc, latent_initial_fc] + biLSTM.get_weights() + decLSTM.get_weights()
    updates = RMSprop(cost = total_loss, params = params)
    f = theano.function(inputs = [sketch, dec_sketch, gaussian_sample], outputs = [reconstruction_loss, x_mean, y_mean, x_variance, y_variance, correlation, p_values, weightings], updates = updates)

    train_file = open('rain.processed')
    train_set = train_file.read()
    train_file.close()
    
    drawings = train_set.split('\n')

    drawing1 = list(eval(drawings[5].replace(' ', ',')))
    drawing2 = list(eval(drawings[1].replace(' ', ',')))

    batch = np.array([drawing1, drawing1]).transpose([1, 0, 2]).astype(float)
    target_d = [[0, 0, 1, 0, 0]] + drawing1[:-1]

    target = np.array([target_d, target_d]).transpose([1, 0, 2]).astype(float)

    batch[:, :, :2] /= 2.
    target[:, :, :2] /= 2.

    print(target.shape, 'target')
    s = np.random.randn(1, 101)

    for i in range (100):
        # We're going to use the exact same gaussian sample. If we did everything correctly, the values should be exactly the same for both batch elements
        print(f(batch, target, np.concatenate([s, s], axis = 0))[0])
       # print(f(batch, target, np.concatenate([s, s], axis = 0))[1][10][:, 1])


main()

    
