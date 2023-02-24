import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.distributions import Normal
import math
import numpy as np

import pyscipopt as scip
from pyscipopt import SCIP_RESULT

from beam_search import Beam
from utils import cut_feature_generator
from logger import logger
from pointer_net import Encoder

LOG_STD_MAX = 2
LOG_STD_MIN = -20

# from ipdb import set_trace
# from ipdb import set_trace 

class Attention(nn.Module):
    """A generic attention module for a decoder in seq2seq"""
    def __init__(self, dim, use_tanh=False, C=10, use_cuda=True):
        super(Attention, self).__init__()
        self.use_tanh = use_tanh
        self.project_query = nn.Linear(dim, dim)
        self.project_ref = nn.Conv1d(dim, dim, 1, 1)
        self.C = C  # tanh exploration
        self.tanh = nn.Tanh()
        
        # v = torch.FloatTensor(dim)
        # if use_cuda:
        #     v = v.cuda()  
        self.v = nn.Parameter(torch.FloatTensor(dim))
        self.v.data.uniform_(-(1. / math.sqrt(dim)) , 1. / math.sqrt(dim))
        
    def forward(self, query, ref):
        """
        Args: 
            query: is the hidden state of the decoder at the current
                time step. batch x dim
            ref: the set of hidden states from the encoder. 
                sourceL x batch x hidden_dim
        """
        # ref is now [batch_size x hidden_dim x sourceL]
        ref = ref.permute(1, 2, 0)
        q = self.project_query(query).unsqueeze(2)  # batch x dim x 1
        e = self.project_ref(ref)  # batch_size x hidden_dim x sourceL 
        # expand the query by sourceL
        # batch x dim x sourceL
        expanded_q = q.repeat(1, 1, e.size(2)) 
        # batch x 1 x hidden_dim
        v_view = self.v.unsqueeze(0).expand(
                expanded_q.size(0), len(self.v)).unsqueeze(1)
        # [batch_size x 1 x hidden_dim] * [batch_size x hidden_dim x sourceL]
        u = torch.bmm(v_view, self.tanh(expanded_q + e)).squeeze(1)
        if self.use_tanh:
            logits = self.C * self.tanh(u)
        else:
            logits = u  
        return e, logits

class DecoderEndToken(nn.Module):
    def __init__(self, 
            embedding_dim,
            hidden_dim,
            tanh_exploration,
            use_tanh,
            n_glimpses=1,
            beam_size=0,
            use_cuda=True):
        super(DecoderEndToken, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_glimpses = n_glimpses
        self.beam_size = beam_size
        self.use_cuda = use_cuda

        self.input_weights = nn.Linear(embedding_dim, 4 * hidden_dim)
        self.hidden_weights = nn.Linear(hidden_dim, 4 * hidden_dim)

        self.pointer = Attention(hidden_dim, use_tanh=use_tanh, C=tanh_exploration, use_cuda=self.use_cuda)
        self.glimpse = Attention(hidden_dim, use_tanh=False, use_cuda=self.use_cuda)
        self.sm = nn.Softmax()

    def apply_mask_to_logits(self, step, logits, mask, prev_idxs):    
        if mask is None:
            mask = torch.zeros(logits.size()).byte().to(self.pointer.v.device)
            # if self.use_cuda:
            #     mask = mask.cuda()
    
        maskk = mask.clone()

        # to prevent them from being reselected. 
        # Or, allow re-selection and penalize in the objective function
        if prev_idxs is not None:
            # set most recently selected idx values to 1
            maskk[[x for x in range(logits.size(0))],
                    prev_idxs.data] = 1
            logits[maskk] = -np.inf
        return logits, maskk

    def logprobs(self, decoder_input, embedded_inputs, hidden, context, max_length, seled_idxes):
        def recurrence(x, hidden, logit_mask, prev_idxs, step):
            
            hx, cx = hidden  # batch_size x hidden_dim
            
            gates = self.input_weights(x) + self.hidden_weights(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)  # batch_size x hidden_dim
            
            g_l = hy
            for i in range(self.n_glimpses):
                ref, logits = self.glimpse(g_l, context)
                logits, logit_mask = self.apply_mask_to_logits(step, logits, logit_mask, prev_idxs)
                # [batch_size x h_dim x sourceL] * [batch_size x sourceL x 1] = 
                # [batch_size x h_dim x 1]
                g_l = torch.bmm(ref, self.sm(logits).unsqueeze(2)).squeeze(2) 
            _, logits = self.pointer(g_l, context) # logits 代表基于context vector 的概率分布
            
            logits, logit_mask = self.apply_mask_to_logits(step, logits, logit_mask, prev_idxs)
            probs = self.sm(logits)
            return hy, cy, probs, logit_mask
    
        batch_size = context.size(1)
        outputs = []
        single_probs = []
        steps = range(max_length)  # or until terminating symbol ?
        inps = []
        idxs = None
        mask = None
       
        for i in steps:
            hx, cx, probs, mask = recurrence(decoder_input, hidden, mask, idxs, i)
            hidden = (hx, cx)
            # select the next inputs for the decoder [batch_size x hidden_dim]
            decoder_input, prob = self.decode_logp(
                probs,
                embedded_inputs,
                seled_idxes[i]) # 每一次decode 都是随机sample 一个输出
            inps.append(decoder_input) 
            idxs = torch.tensor([seled_idxes[i]], dtype=torch.int64).to(self.pointer.v.device)
            # if self.use_cuda:
            #     idxs = idxs.cuda()
            # use outs to point to next object
            outputs.append(probs)
            single_probs.append(prob)

        return (outputs, single_probs), hidden

    def decode_logp(self, probs, embedded_inputs, idxs):
        batch_size = probs.size(0)
        # due to race conditions, might need to resample here
        sels = embedded_inputs[idxs, [i for i in range(batch_size)], :] 
        return sels, probs[:,idxs]

    def forward(self, decoder_input, embedded_inputs, hidden, context, max_length, decode_type):
        """
        Args:
            decoder_input: The initial input to the decoder
                size is [batch_size x embedding_dim]. Trainable parameter.
            embedded_inputs: [sourceL x batch_size x embedding_dim]
            hidden: the prev hidden state, size is [batch_size x hidden_dim]. 
                Initially this is set to (enc_h[-1], enc_c[-1])
            context: encoder outputs, [sourceL x batch_size x hidden_dim] 
        """
        def recurrence(x, hidden, logit_mask, prev_idxs, step):
            
            hx, cx = hidden  # batch_size x hidden_dim
            
            gates = self.input_weights(x) + self.hidden_weights(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)  # batch_size x hidden_dim
            
            g_l = hy
            for i in range(self.n_glimpses):
                ref, logits = self.glimpse(g_l, context)
                logits, logit_mask = self.apply_mask_to_logits(step, logits, logit_mask, prev_idxs)
                # [batch_size x h_dim x sourceL] * [batch_size x sourceL x 1] = 
                # [batch_size x h_dim x 1]
                g_l = torch.bmm(ref, self.sm(logits).unsqueeze(2)).squeeze(2) 
            _, logits = self.pointer(g_l, context) # logits 代表基于context vector 的概率分布
            
            logits, logit_mask = self.apply_mask_to_logits(step, logits, logit_mask, prev_idxs)
            probs = self.sm(logits)
            return hy, cy, probs, logit_mask
    
        batch_size = context.size(1)
        outputs = []
        selections = []
        steps = range(max_length)  # or until terminating symbol ?
        inps = []
        idxs = None
        mask = None
       
        if decode_type in ["stochastic", "greedy"]:
            for i in steps:
                hx, cx, probs, mask = recurrence(decoder_input, hidden, mask, idxs, i)
                hidden = (hx, cx)
                # select the next inputs for the decoder [batch_size x hidden_dim]
                decoder_input, idxs = self.decode(
                    probs,
                    embedded_inputs,
                    selections,
                    decode_type) # 每一次decode 都是随机sample 一个输出
                inps.append(decoder_input) 
                # use outs to point to next object
                outputs.append(probs)
                selections.append(idxs)

                if idxs.item() == (max_length - 1):
                    # choose the last end token
                    break
            return (outputs, selections), hidden
        
        elif decode_type == "beam_search":
            
            # Expand input tensors for beam search
            decoder_input = Variable(decoder_input.data.repeat(self.beam_size, 1))
            context = Variable(context.data.repeat(1, self.beam_size, 1))
            hidden = (Variable(hidden[0].data.repeat(self.beam_size, 1)),
                    Variable(hidden[1].data.repeat(self.beam_size, 1)))
            
            beam = [
                    Beam(self.beam_size, max_length, cuda=self.use_cuda) 
                    for k in range(batch_size)
            ]
            
            for i in steps:
                hx, cx, probs, mask = recurrence(decoder_input, hidden, mask, idxs, i)
                hidden = (hx, cx)
                
                probs = probs.view(self.beam_size, batch_size, -1
                        ).transpose(0, 1).contiguous()
                
                n_best = 1
                # select the next inputs for the decoder [batch_size x hidden_dim]
                decoder_input, idxs, active = self.decode_beam(probs,
                        embedded_inputs, beam, batch_size, n_best, i)
               
                inps.append(decoder_input) 
                # use probs to point to next object
                if self.beam_size > 1:
                    outputs.append(probs[:, 0,:])
                else:
                    outputs.append(probs.squeeze(0))
                # Check for indexing
                selections.append(idxs)
                 # Should be done decoding
                if len(active) == 0:
                    break
                decoder_input = Variable(decoder_input.data.repeat(self.beam_size, 1))

            return (outputs, selections), hidden

        else:
            raise NotImplementedError

    def decode(self, probs, embedded_inputs, selections, decode_type):
        """
        Return the next input for the decoder by selecting the 
        input with sampling

        Args: 
            probs: [batch_size x sourceL]
            embedded_inputs: [sourceL x batch_size x embedding_dim]
            selections: list of all of the previously selected indices during decoding
       Returns:
            Tensor of size [batch_size x sourceL] containing the embeddings
            from the inputs corresponding to the [batch_size] indices
            selected for this iteration of the decoding, as well as the 
            corresponding indicies
        """
        batch_size = probs.size(0)
        # idxs is [batch_size]
        if decode_type == "stochastic":
            idxs = probs.multinomial(num_samples=1).squeeze(1)
        elif decode_type == "greedy":
            max_probs, idxs = probs.max(1)
        assert idxs not in set(selections)

        sels = embedded_inputs[idxs.data, [i for i in range(batch_size)], :] 
        return sels, idxs


    def decode_beam(self, probs, embedded_inputs, beam, batch_size, n_best, step):
        active = []
        for b in range(batch_size):
            if beam[b].done:
                continue

            if not beam[b].advance(probs.data[b]):
                active += [b]
        
        
        all_hyp, all_scores = [], []
        for b in range(batch_size):
            scores, ks = beam[b].sort_best()
            all_scores += [scores[:n_best]]
            hyps = zip(*[beam[b].get_hyp(k) for k in ks[:n_best]])
            all_hyp += [hyps]
        
        all_idxs = Variable(torch.LongTensor([[x for x in hyp] for hyp in all_hyp]).squeeze())
      
        if all_idxs.dim() == 2:
            if all_idxs.size(1) > n_best:
                idxs = all_idxs[:,-1]
            else:
                idxs = all_idxs
        elif all_idxs.dim() == 3:
            idxs = all_idxs[:, -1, :]
        else:
            if all_idxs.size(0) > 1:
                idxs = all_idxs[-1]
            else:
                idxs = all_idxs
        
        # if self.use_cuda:
        #     idxs = idxs.cuda()
        idxs = idxs.to(self.pointer.v.device)

        if idxs.dim() > 1:
            x = embedded_inputs[idxs.transpose(0,1).contiguous().data, 
                    [x for x in range(batch_size)], :]
        else:
            x = embedded_inputs[idxs.data, [x for x in range(batch_size)], :]
        return x.view(idxs.size(0) * n_best, embedded_inputs.size(2)), idxs, active

class PointerNetworkEndToken(nn.Module):
    """The pointer network, which is the core seq2seq 
    model"""
    def __init__(self, 
            embedding_dim,
            hidden_dim,
            n_glimpses,
            tanh_exploration, # tanh exploration coefficient
            use_tanh,
            beam_size,
            use_cuda):
        super(PointerNetworkEndToken, self).__init__()

        self.embedding_dim = embedding_dim
        self.encoder = Encoder(
                embedding_dim,
                hidden_dim,
                use_cuda)

        self.decoder = DecoderEndToken(
                embedding_dim,
                hidden_dim,
                tanh_exploration=tanh_exploration,
                use_tanh=use_tanh,
                n_glimpses=n_glimpses,
                beam_size=beam_size,
                use_cuda=use_cuda)

        # Trainable initial hidden states
        # dec_in_0 = torch.FloatTensor(embedding_dim)
        # if use_cuda:
        #     dec_in_0 = dec_in_0.cuda()

        self.decoder_in_0 = nn.Parameter(torch.FloatTensor(embedding_dim))
        self.decoder_in_0.data.uniform_(-(1. / math.sqrt(embedding_dim)),
                1. / math.sqrt(embedding_dim))
            
    def forward(self, inputs, max_decode_len, decode_type):
        """ Propagate inputs through the network
        Args: 
            inputs: [sourceL x batch_size x embedding_dim]
        """
        # preprocess inputs 
        end_token = torch.ones((1,inputs.shape[1],inputs.shape[2]),dtype=torch.float,device=inputs.device)
        inputs = torch.cat((inputs, end_token), axis=0)

        (encoder_hx, encoder_cx) = self.encoder.enc_init_state
        encoder_hx = encoder_hx.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0)       
        encoder_cx = encoder_cx.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0)       
        
        # encoder forward pass
        enc_h, (enc_h_t, enc_c_t) = self.encoder(inputs, (encoder_hx, encoder_cx))

        dec_init_state = (enc_h_t[-1], enc_c_t[-1])
    
        # repeat decoder_in_0 across batch
        decoder_input = self.decoder_in_0.unsqueeze(0).repeat(inputs.size(1), 1)
        (pointer_probs, input_idxs), dec_hidden_t = self.decoder(decoder_input,
                inputs,
                dec_init_state,
                enc_h,
                max_decode_len,
                decode_type)

        return pointer_probs, input_idxs

    def _prob_to_logp(self, prob):
        logprob = 0
        for p in prob:
            logp = torch.log(p)
            logprob += logp
        # logprob[(logprob < -10000).detach()] = 0.
        
        return logprob

    def logprobs(self, inputs, max_decode_len, seled_idxes):
        """ Propagate inputs through the network
        Args: 
            inputs: [sourceL x batch_size x embedding_dim]
        """
        # preprocess inputs 
        end_token = torch.ones((1,inputs.shape[1],inputs.shape[2]),dtype=torch.float,device=inputs.device)
        inputs = torch.cat((inputs, end_token), axis=0)

        (encoder_hx, encoder_cx) = self.encoder.enc_init_state
        encoder_hx = encoder_hx.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0)       
        encoder_cx = encoder_cx.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0)       
        
        # encoder forward pass
        enc_h, (enc_h_t, enc_c_t) = self.encoder(inputs, (encoder_hx, encoder_cx))

        dec_init_state = (enc_h_t[-1], enc_c_t[-1])
    
        # repeat decoder_in_0 across batch
        decoder_input = self.decoder_in_0.unsqueeze(0).repeat(inputs.size(1), 1)

        (pointer_probs, probs), dec_hidden_t = self.decoder.logprobs(decoder_input,
                inputs,
                dec_init_state,
                enc_h,
                max_decode_len,
                seled_idxes)
        logprob = self._prob_to_logp(probs)
        
        return [pointer_prob.cpu().detach() for pointer_prob in pointer_probs], logprob


    
# test 
if __name__ == "__main__":
    import time
    embedding_dim = 13
    hidden_dim = 128
    max_decoding_len = 100
    decode_type = 'greedy'
    n_glimpses = 2
    tanh_exploration = 10
    use_tanh = True
    use_cuda = True
    beam_size = 1

    ptr_net = PointerNetworkEndToken(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        n_glimpses=n_glimpses,
        tanh_exploration=tanh_exploration,
        use_tanh=use_tanh,
        beam_size=beam_size,
        use_cuda=use_cuda
    ).to('cuda:0')

    # (seq_len, batch_size, feature_dim)
    input_x = torch.randn((max_decoding_len, 1, 13)).to('cuda:0')
    for _ in range(5):
        st = time.time()
        pointer_probs, input_idxs = ptr_net(input_x, max_decoding_len+1, decode_type)
        et = time.time() - st
        _, logp = ptr_net.logprobs(input_x, len(input_idxs), input_idxs)
        print(f"time: {et}")
        print(f"selected_idxs: {len(input_idxs)}")
