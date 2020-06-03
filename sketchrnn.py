import torch
from torch import nn
from torch import optim
from torch import autograd

import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
import pygame
from pygame.locals import *

from rich.progress import track

import numpy as np
import json
import time 

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # need to generate input, forget, cell, output
        self.input_hidden = nn.Linear(self.input_size, self.hidden_size * 4)
        self.hidden_hidden = nn.Linear(self.hidden_size, self.hidden_size * 4)

    def forward(self, inp, state):
        cell, hidden = state

        state = self.input_hidden(inp) + self.hidden_hidden(hidden)

        i, f, c, o = state.split(self.hidden_size, 1)

        input_gate = torch.sigmoid(i)
        forget_gate = torch.sigmoid(f)
        cell_gate = torch.tanh(c)
        output_gate = torch.sigmoid(o)

        cell = forget_gate * cell + input_gate * cell_gate
        hidden = output_gate * cell

        return cell, hidden

    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size ), torch.zeros(bs, self.hidden_size)
    
 
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_Size

        self.forward_lstm = LSTM(self.input_size, self.hidden_size)
        self.backward_lstm = LSTM(self.input_size, self.hidden_size)

    def forward(self, inp, fstate, bstate):
        fcell, fhidden = self.forward_lstm.forward(finp, fstate)
        bcell, bhidden = self.backward_lstm.forward(binp, bstate) 

        return [fcell, fhidden], [bcell, bhidden]

    def init_hidden(self, bs):
        return self.forward_lstm.init_hidden(bs), self.backward_lstm.init_hidden(bs)

class SketchRNNUnconditional(nn.Module):
    def __init__(self, input_size, hidden_size, M = 8):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.decoder = LSTM(input_size, hidden_size)
        self.decoder_output = nn.Linear(self.hidden_size, 5 * M + M + 3)

        self.M = M 

    def forward(self, inp, state):
        decoder_state = self.decoder(inp, state) 

        # only feed the hidden state of the decoder
        outputs = self.decoder_output(decoder_state[1])

        M = self.M

        # meanx -> bs x 6M + 3
        meanx = outputs[:, :M]
        meany = outputs[:, M : 2 * M]
        logvarx = outputs[:, 2 * M : 3 * M]
        logvary = outputs[:, 3 * M : 4 * M]
        correlation = torch.tanh(outputs[:, 4 * M : 5 * M]) * 0.9    # restrict to (-1, 1)
        weights = torch.softmax(outputs[:, 5 * M : 6 * M] + 1e-5 , axis = -1) + 1e-5 

        pen = outputs[:, 6 * M :]
        pen = torch.softmax(pen + 1e-5, axis = -1) + 1e-5

        varx = torch.exp(logvarx)
        vary = torch.exp(logvary)

        return decoder_state, meanx, meany, varx, vary, correlation, weights, pen

    def init_hidden(self, bs):
        return self.decoder.init_hidden(bs)

def generate_dataset(path, parts = 5, max_length = 100):
    dataset = []
    with open(path, 'r') as data:
        lines = data.readlines()

        for l in track(lines, description = "processing..."):
            strokes = []

            d = json.loads(l)
            drawing = d['drawing']
            x, y = drawing[0][0][0], drawing[0][1][0]

            for stroke in drawing:
                xs, ys = stroke
                # add the beginning pen up movement to the start of the next stroke
                strokes.append([
                    xs[0] - x, ys[0] - y, 0
                ])

                x, y = xs[0], ys[0]

                for t in range (1, len(xs)):
                    strokes.append([
                        xs[t] - x, ys[t] - y, 1
                    ])

                    x, y = xs[t], ys[t]
            
            strokes.append([0, 0, 2])     # end drawing
            
            del strokes[0]

            if len(strokes) < max_length:
                dataset.append(strokes)
    
    # go through dataset and pad
    for d in track(range(len(dataset)), description = 'padding...   '):
        if len(dataset[d]) < max_length:
            dataset[d] = dataset[d] + [[0, 0, 2]] * (max_length - len(dataset[d]))

    # save each section of the dataset
    split = len(dataset) // parts
    print("Splitting into %s sections of %s examples each" % (parts, split))
    for p in track(range (parts), description = 'splitting... '):
        section = dataset[ : split]
        del dataset[: split]

        ds = np.array(section)
        np.save('data/cloud_p%s.npy' % p, ds)
        #print(ds[0])

#ds = generate_dataset("full_simplified_cloud.ndjson")

#np.save('cloud.npy', ds, allow_pickle = False)

#ds = np.load('cloud.npy')
#print(ds)

def sample_drawing(model, seed, ts = 100, temperature = 1):
    info = []
    drawing = seed.copy()
    with torch.no_grad():
        test_state = model.init_hidden(1)
        
        # seed model with start
        for s in seed[:-1]:
            pen_inp = [0, 0, 0]
            pen_inp[s[2]] = 1.
            inp = torch.tensor([s[0], s[1]] + pen_inp)
            test_state, _, _, _, _, _, _, _ = model.forward(inp, test_state)

        for t in range (ts):
            pen_inp = [0, 0, 0]
            pen_inp[drawing[-1][2]] = 1.

            inp = torch.tensor([drawing[-1][0], drawing[-1][1]] + pen_inp)

            test_state, meanx, meany, varx, vary, correlation, weights, pen = model.forward(inp, test_state)

            # TODO: Incorporate temperature
            # sample from the mixture model
            #print(weights[0])
            mixture = np.random.choice(range(0, model.M), p = weights[0].numpy())

            mx, my, vx, vy, c = meanx[0, mixture], meany[0, mixture], varx[0, mixture], vary[0, mixture], correlation[0, mixture]
            mx, my, vx, vy, c = [i.numpy().item() for i in [mx, my, vx, vy, c]]
            pen = pen[0].numpy()

            # sample coordinates
            x, y = np.random.multivariate_normal(mean = [mx, my], cov = [[vx, c * np.sqrt(vx * vy)], [c * np.sqrt(vx * vy), vy]])

            # sample pen
            p = np.random.choice([0, 1, 0], p = pen)

            drawing.append([x, y, p])
            info.append([mx, my, vx, vy, c, weights, pen])
    
    return drawing, info 
            
def display_drawing(surface, drawing, offset, color = (130, 90, 230), scale = 20):
    x, y = offset[0] - 250, offset[1] - 250
    surface.fill((255, 255, 255))

    for stroke in drawing:
        if stroke[2] == 1:
            c = color
            if not color:
                c = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 200))

            pygame.draw.line(surface, c, (int(x+ 250), int(y + 250)), (int(x + 250 + stroke[0] * scale), int(y + 250 + stroke[1] * scale)), 2)
        if stroke[2] == 2:
            break;
        
        x += stroke[0] * scale
        y += stroke[1] * scale
        
        #time.sleep(0.05)
    pygame.display.flip()
    
def main():
    # concept, neural cellular automata for generation of sentences
    # give 512 tokens of context, so the image is like V x 1025 x 5 where V is the number of channels and is the vocab size
    # the context tokens are along the middle row of the image. The image is centered around the token to be predicted.
    # the other 512 tokens to the right of the predicted token is additional workspace, hopefully allows network to flesh out future sentences to form the predicted token.
    # network is given X timesteps to form its next character
    # loss is computed on the center pixel's vector which is size V
    # after X timesteps the entire image is shifted to the left, and the next token is predicted.
    # after X * n timesteps we perform BPTT, n starts out at some small number and is linearly increased to some larger number to learn long-term sequences

    # load dataset here
    # all the data is in the form of
    # x, y, pen_up, pen_down, drawing_end
    # we limit all drawings to length of MAX_LENGTH
    # any drawing less than length MAX_LENGTH just pads the rest of the inputs with 0, 0, 0, 0, 1

    # TODO: Add script to check for data and if there is not generate it from ndjson file/download it
    # TODO: Make the decoder a HyperLSTM

    pygame.init()
    screen = pygame.display.set_mode((500, 500))
    done= False 

    test = False

    max_ts = 100
    bs = 32 # should be 32!
    M = 8

    decoder = SketchRNNUnconditional(5, hidden_size = 256, M = M)
    optimizer = optim.Adam(decoder.parameters(), lr = 0.0001)

    for p in decoder.parameters():
        p.register_hook(
            lambda gradient: torch.clamp(gradient, -1, 1)
        )

    decoder.load_state_dict(torch.load("models/sketchRNNUnconditional_0_3.pt" ).state_dict())

    if test:
        seed = []
        offset = []
        c = -1
        x = 0
        y = 0
        d = []
        while not done:
            c += 1
            drawing = False 

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    done = True 
            
                if event.type == pygame.MOUSEBUTTONDOWN:
                    drawing = True

                    pos = pygame.mouse.get_pos()

                    if len(seed) > 0:
                        # get relative position from the previous stroke
                        if seed[-1][2] != 0:
                            seed.append([
                                (pos[0] - x) / 20,
                                (pos[1] - y) / 20,
                                0       # pen up
                            ])

                        x, y = pos

                    else:
                        offset = pos
                        x, y = pos

                elif event.type == pygame.MOUSEMOTION:
                    pos = pygame.mouse.get_pos()

                    if pygame.mouse.get_pressed()[0] and c % 5 == 0:
                        drawing = True 
                        seed.append([
                            (pos[0] - x) / 20,
                            (pos[1] - y) / 20,
                            1           # pen down
                        ])
                
                        x, y = pos

            
            if seed and not drawing:
                if c % 10 == 0:
                    d, info = sample_drawing(decoder, seed)
                    display_drawing(screen, d, offset, scale = 20)
            
            elif seed:
                display_drawing(screen, seed, offset, scale = 20)
            

            pygame.display.flip()

            time.sleep(1/60.)       # cap to 60 fps
    

    with autograd.detect_anomaly():
        for epoch in range (1, 100):
            # data loop
            for p in range (5):
                torch.save(decoder, "models/sketchRNNUnconditional_%s_%s.pt" % (epoch, p))

                ds = np.load('data/cloud_p%s.npy' % p)

                dl = DataLoader(dataset = ds, batch_size = bs, shuffle = True, drop_last = True)

                for X in dl:
                    for t in range (1):
                        strokeTargets = torch.transpose(torch.tensor(X[:,:,:2]).type(torch.FloatTensor) / 10.,
                                            0, 1)

                        penTargets = torch.transpose(torch.tensor(X[:, :, 2:]), 0, 1).type(torch.LongTensor)

                        penTargets = torch.zeros(max_ts, bs, 3).scatter_(
                            2, penTargets, 1.
                        )

                        #print(penTargets)

                        state = decoder.init_hidden(bs)

                        loss_pen = 0
                        loss_stroke = 0

                        recording_stroke_loss = True

                        d = [] 

                        for ts in range (max_ts - 1):
                            # inp -> bs x 2 + bs x 3 => bs x 5
                            inp = torch.cat([strokeTargets[ts].detach(), penTargets[ts].detach()], axis = -1)

                            # s, x, y, vx, vy, c, weights -> bs x M
                            # pen -> bs x 3
                            state, meanx, meany, varx, vary, correlation, weights, pen = decoder.forward(
                                inp, state
                            )

                            # compute loss for pen, which is computed across all timesteps
                            loss_pen -= torch.mean(
                                torch.log(torch.sum(penTargets[ts + 1].detach() * pen, axis = -1))    # log -> bs
                            )

                            # compute loss for stroke, which is only computed until the sketch ends
                            if np.argmax(penTargets[ts]) == 2:
                                recording_stroke_loss = False

                            if recording_stroke_loss:
                                # the loss is the probability of the correct stroke offset from the gaussian mixture
                                # this is a weighted average of the PDFS of each of the gaussians
                                # the pdf of a 2d multivariate gaussian with a covariance matrix is
                                # (2pi)^[-1] * det(COV)^-0.5 * e^(-0.5 * (x - mean).T * COV^-1 * (x - mean))

                                # COV = varx           correlation
                                #       correlation    vary

                                # COV -> 2 x 2 x bs x M, transpose -> bs x M x 2 x 2

                                # det(COV) = varx * vary - correlation ** 2 -> bs x M
                                # x -> bs x 1 (added) x 2, mean -> bs x M x 1 (added) x 2, x - mean -> bs x M x 1 (added) x 2
                                # COV^-1 = 1/det(COV) * -vary         -correlation
                                #                       -correlation  varx = torch.inverse(COV)
                                """meanx = torch.FloatTensor([[1, 2, 3,], [2, 3, 4]])
                                meany = torch.FloatTensor([[-1, -2, -3], [-2, -3, -4]])
                                varx = torch.FloatTensor([[0.1, 1, 10], [0.1, 1, 10]])
                                vary = torch.FloatTensor([[0.1, 1, 3], [0.1, 2, 3]])
                                correlation = torch.FloatTensor([[-0.5, 0, 0.5], [0.5, 0, -0.5]])
                                """ # test values
                                # i am like 99% sure the algorithm below calculates the correct pdf, confirmed it with a few test cases
                                # i could use torch.distributions.multivariate_normal.MultivariateNormal, but where's the fun in that

                                det = varx * vary * (1 - correlation ** 2)

                                # strokeTargets[ts + 1] -> bs x 2
                                # strokeTargets[ts + 1, :, 0:1] -> bs x 1 
                                # x_mean -> bs x 1

                                x_mean = strokeTargets[ts + 1, :, 0:1].detach() - meanx
                                y_mean = strokeTargets[ts + 1, :, 1:].detach() - meany

                                c = correlation * torch.sqrt(varx) * torch.sqrt(vary)
                                exp = -0.5 * 1/det * ((x_mean ** 2 * vary) - c * x_mean * y_mean - c * x_mean * y_mean + varx * y_mean ** 2)
                                
                                pdf = 1/(2 * np.pi) * det ** -0.5 * np.e ** exp + 1e-5    # -> bs x M

                                # compute weighted average of the pdfs
                                # weights -> bs x M, softmax along axis = -1
                                weighted_pdf = torch.sum(pdf * weights, axis = -1, keepdim = True) # -> bs x 1

                                log_probs = torch.log(weighted_pdf)
                                if (torch.isnan(log_probs).any()):
                                    print(pdf, weights, "weighted pdf")

                                loss_stroke -= torch.mean(log_probs)

                                #print(mean.shape, x_mean.shape, det.shape, cov.shape, cov_inverse.shape)

                                #loss_stroke += torch.mean(
                                #    1/(2 * np.pi) * det ** (-0.5) * np.e ** (-0.5 * (torch.dot(x_mean.transpose(3, 4), ))
                                #
                                #print(log_probs)
                            d.append([torch.sum(weights * meanx, axis = -1)[0].detach().numpy().item() * 10, torch.sum(weights * meany, axis = -1)[0].detach().numpy().item() * 10,            np.random.choice([0, 1, 2], p = pen[0].detach().numpy())])

                        
                        loss_pen = 1/max_ts * loss_pen
                        loss_stroke = 1/max_ts * loss_stroke

                        loss = loss_pen + loss_stroke

                        print("loss: %s" % (loss.detach().item()))

                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                        #drawing, info = sample_drawing(decoder)
                        display_drawing(screen, d, offset = [250, 250], scale = 1, color = False)

        """
        
        """ # for displaying a drawing
        
        #if done:
        #    break;

        #for i in range (0):
        #    for ts in range (max_ts):
        #        pass

if __name__ == "__main__":
    main()
