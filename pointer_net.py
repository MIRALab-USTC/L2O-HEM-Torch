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

# from beam_search import Beam
from utils import cut_feature_generator
from logger import logger

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class Encoder(nn.Module):
    """Maps a graph represented as an input sequence
    to a hidden vector"""
    def __init__(self, input_dim, hidden_dim, use_cuda):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim) # default layers=1
        self.use_cuda = use_cuda
        self.enc_init_hx = nn.Parameter(torch.zeros(hidden_dim),requires_grad=False)
        self.enc_init_cx = nn.Parameter(torch.zeros(hidden_dim),requires_grad=False)

        self.enc_init_state = (self.enc_init_hx, self.enc_init_cx)

    def forward(self, x, hidden):
        output, hidden = self.lstm(x, hidden)
        return output, hidden
    
    def init_hidden(self, hidden_dim):
        """Trainable initial hidden state"""
        enc_init_hx = nn.Parameter(torch.zeros(hidden_dim),requires_grad=False)
        # enc_init_hx = Variable(torch.zeros(hidden_dim), requires_grad=False)
        # if self.use_cuda:
        #     enc_init_hx = enc_init_hx.cuda()

        # enc_init_hx.data.uniform_(-(1. / math.sqrt(hidden_dim)),
        #        1. / math.sqrt(hidden_dim))

        enc_init_cx = nn.Parameter(torch.zeros(hidden_dim),requires_grad=False)
        # enc_init_cx = Variable(torch.zeros(hidden_dim), requires_grad=False)
        # if self.use_cuda:
        #     enc_init_cx = enc_init_cx.cuda()

        #enc_init_cx = nn.Parameter(enc_init_cx)
        # enc_init_cx.data.uniform_(-(1. / math.sqrt(hidden_dim)),
        #        1. / math.sqrt(hidden_dim))
        return (enc_init_hx, enc_init_cx)

class Attention(nn.Module):
    """A generic attention module for a decoder in seq2seq"""
    def __init__(self, dim, use_tanh=False, C=10, use_cuda=True):
        super(Attention, self).__init__()
        self.use_tanh = use_tanh
        self.project_query = nn.Linear(dim, dim)
        self.project_ref = nn.Conv1d(dim, dim, 1, 1) # TODO: check 为何会有卷积
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

# TODO：保留beam search，我们的setting 用不了beam search，还得增加一个模式是sample 的模式 
class Decoder(nn.Module):
    def __init__(self, 
            embedding_dim,
            hidden_dim,
            tanh_exploration,
            use_tanh,
            n_glimpses=1,
            beam_size=0,
            use_cuda=True):
        super(Decoder, self).__init__()
        
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
        # print(f"debug log maskk device: {maskk.device}")

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

    def forward(self, decoder_input, embedded_inputs, hidden, context, max_length, decode_type): # TODO: max decode length 以参数传入forward 函数
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
            return (outputs, selections), hidden
        
        elif decode_type == "beam_search":
            raise NotImplementedError
            # Expand input tensors for beam search
            # decoder_input = Variable(decoder_input.data.repeat(self.beam_size, 1))
            # context = Variable(context.data.repeat(1, self.beam_size, 1))
            # hidden = (Variable(hidden[0].data.repeat(self.beam_size, 1)),
            #         Variable(hidden[1].data.repeat(self.beam_size, 1)))
            
            # beam = [
            #         Beam(self.beam_size, max_length, cuda=self.use_cuda) 
            #         for k in range(batch_size)
            # ]
            
            # for i in steps:
            #     hx, cx, probs, mask = recurrence(decoder_input, hidden, mask, idxs, i)
            #     hidden = (hx, cx)
                
            #     probs = probs.view(self.beam_size, batch_size, -1
            #             ).transpose(0, 1).contiguous()
                
            #     n_best = 1
            #     # select the next inputs for the decoder [batch_size x hidden_dim]
            #     decoder_input, idxs, active = self.decode_beam(probs,
            #             embedded_inputs, beam, batch_size, n_best, i)
               
            #     inps.append(decoder_input) 
            #     # use probs to point to next object
            #     if self.beam_size > 1:
            #         outputs.append(probs[:, 0,:])
            #     else:
            #         outputs.append(probs.squeeze(0))
            #     # Check for indexing
            #     selections.append(idxs)
            #      # Should be done decoding
            #     if len(active) == 0:
            #         break
            #     decoder_input = Variable(decoder_input.data.repeat(self.beam_size, 1))

            # return (outputs, selections), hidden

        else:
            # TODO: 实现每轮输出最大概率对应的index
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
            idxs = probs.multinomial(num_samples=1).squeeze(1) # TODO: multinomial() 函数添加参数num_samples，应该也是torch 版本的问题
        elif decode_type == "greedy":
            max_probs, idxs = probs.max(1)
        # due to race conditions, might need to resample here
        # TODO: check，这里的mask 操作是O(n) 感觉不够快
        # for old_idxs in selections:
        #     # compare new idxs
        #     # elementwise with the previous idxs. If any matches,
        #     # then need to resample
        #     if old_idxs.eq(idxs).data.any():
        #         print(' [!] resampling due to race condition')
        #         if decode_type == "stochastic":
        #             idxs = probs.multinomial(num_samples=1).squeeze(1) # TODO: multinomial() 函数添加参数num_samples，应该也是torch 版本的问题
        #         elif decode_type == "greedy":
        #             max_probs, idxs = probs.max(1)
        #         break
        assert idxs not in set(selections)

        sels = embedded_inputs[idxs.data, [i for i in range(batch_size)], :] 
        return sels, idxs


    # def decode_beam(self, probs, embedded_inputs, beam, batch_size, n_best, step):
    #     active = []
    #     for b in range(batch_size):
    #         if beam[b].done:
    #             continue

    #         if not beam[b].advance(probs.data[b]):
    #             active += [b]
        
        
    #     all_hyp, all_scores = [], []
    #     for b in range(batch_size):
    #         scores, ks = beam[b].sort_best()
    #         all_scores += [scores[:n_best]]
    #         hyps = zip(*[beam[b].get_hyp(k) for k in ks[:n_best]])
    #         all_hyp += [hyps]
        
    #     all_idxs = Variable(torch.LongTensor([[x for x in hyp] for hyp in all_hyp]).squeeze())
      
    #     if all_idxs.dim() == 2:
    #         if all_idxs.size(1) > n_best:
    #             idxs = all_idxs[:,-1]
    #         else:
    #             idxs = all_idxs
    #     elif all_idxs.dim() == 3:
    #         idxs = all_idxs[:, -1, :]
    #     else:
    #         if all_idxs.size(0) > 1:
    #             idxs = all_idxs[-1]
    #         else:
    #             idxs = all_idxs
        
    #     # if self.use_cuda:
    #     #     idxs = idxs.cuda()
    #     idxs = idxs.to(self.pointer.v.device)

    #     if idxs.dim() > 1:
    #         x = embedded_inputs[idxs.transpose(0,1).contiguous().data, 
    #                 [x for x in range(batch_size)], :]
    #     else:
    #         x = embedded_inputs[idxs.data, [x for x in range(batch_size)], :]
    #     return x.view(idxs.size(0) * n_best, embedded_inputs.size(2)), idxs, active

class PointerNetwork(nn.Module):
    """The pointer network, which is the core seq2seq 
    model"""
    def __init__(self, 
            embedding_dim,
            hidden_dim,
            n_glimpses, # TODO：这个参数代表什么意思？
            tanh_exploration, # tanh exploration coefficient
            use_tanh,
            beam_size,
            use_cuda):
        super(PointerNetwork, self).__init__()

        self.embedding_dim = embedding_dim
        self.encoder = Encoder(
                embedding_dim,
                hidden_dim,
                use_cuda)

        self.decoder = Decoder(
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
        # TODO: 添加截断过小logprob 的trick 
        # logprob[(logprob < -10000).detach()] = 0.
        
        return logprob

    def logprobs(self, inputs, max_decode_len, seled_idxes):
        """ Propagate inputs through the network
        Args: 
            inputs: [sourceL x batch_size x embedding_dim]
        """
        
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

class CriticNetwork(nn.Module):
    """Useful as a baseline in REINFORCE updates"""
    def __init__(self,
            embedding_dim,
            hidden_dim,
            n_process_block_iters,
            tanh_exploration,
            use_tanh,
            use_cuda):
        super(CriticNetwork, self).__init__()
        # TODO: check embedding_dim 是否还有必要呢？
        self.hidden_dim = hidden_dim
        self.n_process_block_iters = n_process_block_iters

        self.encoder = Encoder(
                embedding_dim,
                hidden_dim,
                use_cuda)
        
        self.process_block = Attention(hidden_dim,
                use_tanh=use_tanh, C=tanh_exploration, use_cuda=use_cuda)
        self.sm = nn.Softmax()
        self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
        )

    def forward(self, inputs):
        """
        Args:
            inputs: [embedding_dim x batch_size x sourceL] of embedded inputs
        """
         
        (encoder_hx, encoder_cx) = self.encoder.enc_init_state
        encoder_hx = encoder_hx.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0)
        encoder_cx = encoder_cx.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0)       
        
        # encoder forward pass
        enc_outputs, (enc_h_t, enc_c_t) = self.encoder(inputs, (encoder_hx, encoder_cx))
        
        # grab the hidden state and process it via the process block 
        process_block_state = enc_h_t[-1]
        for i in range(self.n_process_block_iters):
            ref, logits = self.process_block(process_block_state, enc_outputs)
            process_block_state = torch.bmm(ref, self.sm(logits).unsqueeze(2)).squeeze(2)
        # produce the final scalar output
        out = self.decoder(process_block_state)
        return out

class CutsPercentPolicy(nn.Module):
    def __init__(self,
            embedding_dim,
            hidden_dim,
            n_process_block_iters,
            tanh_exploration,
            use_tanh,
            use_cuda):
        super(CutsPercentPolicy, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_process_block_iters = n_process_block_iters

        self.encoder = Encoder(
                embedding_dim,
                hidden_dim,
                use_cuda)
        
        self.process_block = Attention(hidden_dim,
                use_tanh=use_tanh, C=tanh_exploration, use_cuda=use_cuda)
        self.sm = nn.Softmax()
        self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2)
        )
        self.use_cuda = use_cuda

    def forward(self, inputs):
        """
        Args:
            inputs: [embedding_dim x batch_size x sourceL] of embedded inputs
        """
         
        (encoder_hx, encoder_cx) = self.encoder.enc_init_state
        encoder_hx = encoder_hx.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0)
        encoder_cx = encoder_cx.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0)       
        
        # encoder forward pass
        enc_outputs, (enc_h_t, enc_c_t) = self.encoder(inputs, (encoder_hx, encoder_cx))
        
        # grab the hidden state and process it via the process block 
        process_block_state = enc_h_t[-1]
        for i in range(self.n_process_block_iters):
            ref, logits = self.process_block(process_block_state, enc_outputs)
            process_block_state = torch.bmm(ref, self.sm(logits).unsqueeze(2)).squeeze(2)
        # produce the final scalar output
        out = self.decoder(process_block_state)
        return out

    def action(self, states, deterministic=False):
        mean, log_std = self.get_mean_std(states)
        std = torch.exp(log_std)
        # normal = Normal(mean, std)
        if deterministic:
            action = mean
        else:
            if self.use_cuda:
                sample = Normal(torch.zeros_like(mean).to(states.device), torch.ones_like(mean).to(states.device)).sample()
            else:
                sample = Normal(torch.zeros_like(mean), torch.ones_like(mean)).sample()
            action = mean + std * sample
        
        tanh_action = torch.tanh(action)

        return tanh_action

    def get_mean_std(self, states):
        out = self.forward(states)
        mean, log_std = torch.chunk(out,2,-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        log_std = log_std.expand(mean.shape)

        return mean, log_std

    def log_prob(self, states, action=None, pretanh_action=None):
        if pretanh_action is None:
            assert action is not None
            pretanh_action = torch.log((1+action)/(1-action) +1e-6) / 2
        else:
            assert pretanh_action is not None
            action = torch.tanh(pretanh_action)
        mean, log_std = self.get_mean_std(states)
        std = torch.exp(log_std)
        normal = Normal(mean, std)
        pre_log_prob = normal.log_prob(pretanh_action)
        log_prob = pre_log_prob.sum(-1, keepdim=True) - torch.log(1 - action * action + 1e-6).sum(-1, keepdim=True)
        info = {}
        info['pre_log_prob'] = pre_log_prob
        info['mean'] = mean
        info['std'] = std
        info['entropy'] = normal.entropy()

        return log_prob, info 

class CutSelectAgent(scip.Cutsel):
    def __init__(
        self,
        scip_model,
        pointer_net,
        value_net,
        sel_cuts_percent,
        device,
        decode_type,
        baseline_type
    ):
        super().__init__()
        self.scip_model = scip_model
        self.policy = pointer_net
        self.value = value_net
        self.sel_cuts_percent = sel_cuts_percent
        self.device = device
        self.decode_type = decode_type
        self.baseline_type = baseline_type

        self.data = {}

    def cutselselect(self, cuts, forcedcuts, root, maxnselectedcuts):
        '''first method called in each iteration in the main solving loop. '''
        # this method needs to be implemented by the user
        logger.log(f"forcedcuts length: {len(forcedcuts)}")
        logger.log(f"len cuts: {len(cuts)}")
        num_cuts = len(cuts)
        if num_cuts <= 1:
            return {
                'cuts': cuts, # selected sorted cuts
                'nselectedcuts': 1, # num of selected cuts
                'result': SCIP_RESULT.SUCCESS
            }            
        sel_cuts_num = min(int(num_cuts * self.sel_cuts_percent), int(maxnselectedcuts))
        sel_cuts_num = max(sel_cuts_num, 2)
        cuts_features = cut_feature_generator(self.scip_model, cuts)
        input_cuts = torch.from_numpy(cuts_features).to(self.device)
        input_cuts = input_cuts.reshape(input_cuts.shape[0], 1, input_cuts.shape[1])
        if self.decode_type == 'greedy': # evaluate disable gradient calculation
            with torch.no_grad():
                pointer_probs, input_idxs =  self.policy(input_cuts.float(), sel_cuts_num, self.decode_type) # (list of tensor, list of tensor)
            baseline_value = 0.
        else:
            pointer_probs, input_idxs =  self.policy(input_cuts.float(), sel_cuts_num, self.decode_type) # (list of tensor, list of tensor)
            if self.baseline_type == 'net':
                baseline_value = self.value(input_cuts.float())
            else:
                baseline_value = 0.
        idxes = [input.cpu().detach().item() for input in input_idxs]
        assert len(set(idxes))==len(idxes) # 保证选择的idxes 没有重复的！
        all_idxes = list(range(num_cuts))
        not_sel_idxes = list(set(all_idxes).difference(idxes))
        sorted_cuts = [cuts[idx] for idx in idxes]
        not_sel_cuts = [cuts[n_idx] for n_idx in not_sel_idxes]
        sorted_cuts.extend(not_sel_cuts)
        # debug
        # sorted_cuts = cuts
        self.data = {
            "raw_cuts": cuts_features,
            "len_raw_cuts": num_cuts,
            "selected_idx": idxes,
            "pointer_probs": [prob[:,idx] for prob, idx in zip(pointer_probs, idxes)],
            "raw_seq_pointer_probs": [pointer_prob.cpu().detach() for pointer_prob in pointer_probs],
            "baseline_value": baseline_value
        }

        return {
            'cuts': sorted_cuts, # selected sorted cuts
            'nselectedcuts': sel_cuts_num, # num of selected cuts
            'result': SCIP_RESULT.SUCCESS
        }

    def _get_lp_info(self):
        lp_info = {}
        lp_info['lp_solution_value'] = self.scip_model.getLPObjVal()
        cols = self.scip_model.getLPColsData()
        lp_info['lp_solution_integer_var_value'] = [col.getPrimsol() for col in cols if col.isIntegral()]

        return lp_info

    def get_data(self):
        return self.data

    def free_problem(self):
        self.scip_model.freeProb()
    
# test 
if __name__ == "__main__":
    import time
    # from ipdb import set_trace
    # embedding_dim = 13
    # hidden_dim = 128
    # n_process_block_iters = 3
    # tanh_exploration = 10
    # use_tanh = True
    # use_cuda = True
    # cutsel_policy = CutsPercentPolicy(
    #     embedding_dim=embedding_dim,
    #     hidden_dim=hidden_dim,
    #     n_process_block_iters=n_process_block_iters,
    #     tanh_exploration=tanh_exploration,
    #     use_tanh=use_tanh,
    #     use_cuda=use_cuda
    # ).to('cuda:5')

    # input_x = torch.randn((1000, 1, 13)).to('cuda:5')
    # for _ in range(20):
    #     st = time.time()
    #     y = cutsel_policy.action(input_x)
    #     logp, info = cutsel_policy.log_prob(input_x, action=y.detach())
    #     et = time.time() - st
    #     print(f"pred: {y}")
    #     print(f"lop: {logp}")
    #     for k in info.keys():
    #         print(f"{k}: {info[k]}")
    #     print(f"time: {et}")
    embedding_dim = 13
    hidden_dim = 128
    max_decoding_len = 40
    decode_type = 'greedy'
    n_glimpses = 2
    tanh_exploration = 10
    use_tanh = True
    use_cuda = True
    beam_size = 1

    ptr_net = PointerNetwork(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        n_glimpses=n_glimpses,
        tanh_exploration=tanh_exploration,
        use_tanh=use_tanh,
        beam_size=beam_size,
        use_cuda=use_cuda
    ).to('cuda:4')

    # (seq_len, batch_size, feature_dim)
    input_x = torch.randn((40, 1, 13)).to('cuda:4')
    for _ in range(10):
        st = time.time()
        with torch.no_grad():
            decode_y = ptr_net(input_x, max_decoding_len,'stochastic')
        et = time.time() - st

        print(f"time: {et}")
