import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def cudait(x):
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x

class MultiGRU(nn.Module):
    """ Implements a three layer GRU cell including an embedding layer
       and an output linear layer back to the size of the vocabulary"""
    def __init__(self, voc_size):
        super(MultiGRU, self).__init__()
        self.embedding = nn.Embedding(voc_size, 512) 
        self.rnn = nn.GRU(512, 1024, num_layers=3, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(2048, voc_size)

    def forward(self, x, h):
        x = self.embedding(x)
        h_out = torch.zeros_like(h)
        x, h_out = self.rnn(x, h)
        x = self.linear(x)
        return x, h_out

    def init_h(self, batch_size):
        # Initial cell state is zero
        return cudait(torch.zeros(2*3, batch_size, 1024))

class RNN():
    """Implements the Prior and Agent RNN. Needs a Vocabulary instance in
    order to determine size of the vocabulary and index of the END token"""
    def __init__(self, voc):
        self.rnn = MultiGRU(voc.vocab_size)
        if torch.cuda.is_available():
            self.rnn.cuda()
        self.voc = voc

    def likelihood(self, target): 
        """
            Retrieves the likelihood of a given sequence

            Args:
                target: (batch_size * sequence_lenght) A batch of sequences

            Outputs:
                log_probs : (batch_size) Log likelihood for each example*
        """
        batch_size, seq_length = target.size()
        start_token = cudait(torch.zeros(batch_size, 1).long())
        start_token[:] = self.voc.vocab['GO']
        x = torch.cat((start_token, target[:, :-1]), 1)
        h = self.rnn.init_h(batch_size)
        log_probs = cudait(torch.zeros(batch_size))
        for step in range(seq_length):
            logits, h = self.rnn(x[:, step].view(batch_size, 1), h)
            log_prob = F.log_softmax(logits, dim=2)
            log_probs += NLLLoss(log_prob, target[:, step].view(-1, 1, 1))
        return log_probs

    def sample(self, batch_size, max_length=140):
        """
            Sample a batch of sequences

            Args:
                batch_size : Number of sequences to sample 
                max_length:  Maximum length of the sequences

            Outputs:
                seqs: (batch_size, seq_length) The sampled sequences.
                log_probs : (batch_size) Log likelihood for each sequence.
        """
        start_token = cudait(torch.zeros(batch_size).long())
        start_token[:] = self.voc.vocab['GO']
        h = self.rnn.init_h(batch_size)
        x = start_token

        sequences = []
        log_probs = cudait(torch.zeros(batch_size))
        finished = torch.zeros(batch_size).byte()
        if torch.cuda.is_available():
            finished = finished.cuda()

        for step in range(max_length):
            logits, h = self.rnn(x.view(batch_size, 1), h)
            prob = F.softmax(logits, dim=2)
            log_prob = F.log_softmax(logits, dim=2)
            print(logits.shape)
            x = torch.multinomial(torch.squeeze(prob), 1)
            sequences.append(x.view(-1, 1))
            log_probs +=  NLLLoss(log_prob, x.view(-1, 1, 1))

            x = x.data
            EOS_sampled = (x == self.voc.vocab['EOS']).data
            finished = torch.ge(finished + EOS_sampled, 1)
            if torch.prod(finished) == 1: break

        sequences = torch.cat(sequences, 1)
        return sequences.data, log_probs
    
    def proba(self, batch_size, max_length=140):
        """
            Similar to sample method above. Instead of return the sequence 
            and the NLLL loss. This function will return the probability of 
            each token. This is used for the multi-agent decision making
            scenario.
            
            Args:
                batch_size : Number of sequences to sample 
                max_length:  Maximum length of the sequences
            
            Outputs:
                prob: The probability of each token
        """
        start_token = torch.zeros(batch_size).long()
        start_token[:] = self.voc.vocab['GO']
        h = self.rnn.init_h(batch_size)
        x = start_token
        
        probas = []
        for step in range(max_length):
            logits, h = self.rnn(x.view(batch_size, 1), h)
            prob = F.softmax(logits, dim=2)
            probas.append(prob)
            
        probas = torch.stack(probas)
        probas = torch.squeeze(probas)
        return probas.data

def NLLLoss(inputs, targets):
    """
        Custom Negative Log Likelihood loss that returns loss per example,
        rather than for the entire batch.

        Args:
            inputs : (batch_size, num_classes) *Log probabilities of each class*
            targets: (batch_size) *Target class index*

        Outputs:
            loss : (batch_size) *Loss for each example*
    """

    if torch.cuda.is_available():
        target_expanded = torch.zeros(inputs.size()).cuda()
    else:
        target_expanded = torch.zeros(inputs.size())
    target_expanded.scatter_(2, targets.contiguous().data, 1.0)
    loss = target_expanded * inputs
    loss = torch.sum(loss, 2)
    loss = torch.squeeze(loss)
    return loss
