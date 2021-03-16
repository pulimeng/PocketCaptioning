import torch
import torch.nn as nn
import torch.nn.functional as F

def cudait(x):
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x

class Generator(nn.Module):

    def __init__(self, voc, input_feature_size=512, embed_dim=128, hidden_dim=512,
                 num_layers=3):
            
        super(Generator, self).__init__()
        self.voc = voc
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.features_linear = nn.Sequential(
            nn.Linear(input_feature_size, hidden_dim),
            nn.ReLU()
            )
        self.embedding = nn.Embedding(len(voc), embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers,
                          bidirectional=True, batch_first=True)
        self.output_linear = nn.Linear(2 * hidden_dim, len(voc))
        
    def init_h(self, input_feature):
        hidden = self.features_linear(input_feature.unsqueeze(0))
        hidden = hidden.repeat(2 * self.num_layers, 1, 1)
        return cudait(hidden)
    
    def forward(self, input_feature, sequence):
        inputs = self.features_linear(input_feature)
        h = self.init_h(inputs)
        sequence = self.embedding(sequence)
        x, h_out = self.rnn(sequence, h)
        x = self.output_linear(x)
        return x, h_out
        
    def likelihood(self, input_feature, sequence):
        """
            Retrieves the likelihood of a given sequence
​
            Args:
                input_feature: input voxel features (batch_size * c * w * h * d)
                sequence: (batch_size * sequence_lenght) A batch of sequences
​
            Outputs:
                log_probs : (batch_size) Log likelihood for each example*
        """
        batch_size, seq_length = sequence.size()
        start_token = cudait(torch.zeros(batch_size, 1).long())
        start_token[:] = self.voc.vocab['GO']
        x = torch.cat((start_token, sequence[:, :-1]), 1)
        
        log_probs = cudait(torch.zeros(batch_size))
        
        h = self.init_h(input_feature)
        for step in range(seq_length):
            inputs = self.embedding(x[:, step].view(batch_size, 1))
            logits, h = self.rnn(inputs, h)
            logits = self.output_linear(logits)
            log_prob = F.log_softmax(logits, dim=2)
            log_probs += NLLLoss(log_prob, sequence[:, step].view(-1, 1, 1))
        return log_probs
    
    def sample(self, input_feature, max_length=140):
        """
            Sample a batch of sequences

            Args:
                input_feature : input voxels features (batch_size * c * w * h * d)
                max_length:  Maximum length of the sequences

            Outputs:
                seqs: (batch_size, seq_length) The sampled sequences.
                log_probs : (batch_size) Log likelihood for each sequence.
        """
        batch_size = input_feature.size()[0]
        start_token = cudait(torch.zeros(batch_size).long())
        start_token[:] = self.voc.vocab['GO']
        
        h = self.init_h(input_feature)
        x = start_token

        sequences = []
        log_probs = cudait(torch.zeros(batch_size))
        finished = cudait(torch.zeros(batch_size).byte())
        
        for step in range(max_length):
            inputs = self.embedding(x.view(batch_size, 1))
            logits, h = self.rnn(inputs, h)
            logits = self.output_linear(logits)
            prob = F.softmax(logits, dim=2)
            log_prob = F.log_softmax(logits, dim=2)
            x = torch.multinomial(torch.squeeze(prob), 1)
            sequences.append(x.view(-1, 1))
            log_probs +=  NLLLoss(log_prob, x.view(-1, 1, 1))

            x = x.data
            EOS_sampled = (x == self.voc.vocab['EOS']).data
            finished = torch.ge(finished + EOS_sampled, 1)
            if torch.prod(finished) == 1: break

        sequences = torch.cat(sequences, 1)
        return sequences.data, log_probs
    
    def proba(self, vox_features, max_length=140):
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
        batch_size = vox_features.size()[0]
        start_token = torch.zeros(batch_size).long()
        start_token[:] = self.voc.vocab['GO']
        h = self.init_h(vox_features)
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