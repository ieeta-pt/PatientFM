import torch
from torch import nn
from torchcrf import CRF


r""" The Decoder implements two tasks: I2B2 entity classification, and novel detection of entities. 
     I2B2 part uses the CRF, novel part uses linear+softmax """

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, max_len, batch_size=1, num_layers = 1, bi=True):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.max_len = max_len
        self.mutihead_attention = nn.MultiheadAttention(self.input_size, num_heads=2)

        r""" I2B2 task: CRF """
        self.crf_linear = nn.Linear(self.hidden_size*2, self.output_size)
        self.crf = CRF(self.output_size, batch_first=True)

        r""" Entity detection task: linear + softmax """
        self.entity_linear = nn.Linear(self.hidden_size*2, 2)
        self.softmax = nn.LogSoftmax(dim=2)


    def generate_masked_labels(self, observed_labels, mask, device):
        masked_labels = torch.zeros((mask.size(0), mask.size(1)), dtype=torch.long).to(device)
        for i in range(mask.size(0)):
            masked_labels[i, :len(observed_labels[i][0])] = observed_labels[i][0]

        return masked_labels


    r""" CHECK IF hn IS NECESSARY IN INPUT """
    # def forward(self, encoder_outputs, hn, batch_classes, mask, device):
    def forward(self, encoder_outputs, batch_classes, mask, device):
        r""" Attention block """
        x = encoder_outputs.permute(1, 0, 2)
        attn_output, attn_output_weights = self.mutihead_attention(x, x, x)
        z = attn_output.permute(1, 0, 2)
        decoder_inputs = nn.functional.relu(z)

        r""" I2B2 task: CRF """
        fc_out = self.crf_linear(decoder_inputs)
        #fc_out = self.crf_linear(encoder_outputs)
        masked_labels = self.generate_masked_labels(batch_classes, mask, device)
        mask = mask.type(torch.uint8).to(device)
        crf_loss = self.crf(fc_out, masked_labels, mask, reduction='token_mean')
        crf_out = self.crf.decode(fc_out)

        r""" Entity detection task: linear + softmax """
        seg_weights = self.entity_linear(decoder_inputs)
        seg_out = self.softmax(seg_weights)

        return crf_out, seg_out, -crf_loss
