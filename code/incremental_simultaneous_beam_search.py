import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import search, utils
from fairseq.data import data_utils
from fairseq.models import FairseqIncrementalDecoder
from fairseq.models.fairseq_encoder import EncoderOut
from torch import Tensor
import time, gc

def memReport():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())

class IncrementalSimultaneousBeamSearch:

    def __init__(self,
        model,
        tgt_dict,
        search_strategy,
        beam_size=4,
        length_penalty_alpha=1.0,
        search_mode="classic_nmt",
        encoder_returns="EncoderOut"):

        self.model = model
        self.model.eval()
        self.tgt_dict = tgt_dict
        self.beam_size = beam_size
        self.length_penalty_alpha = length_penalty_alpha
        self.search_mode = search_mode
        assert search_mode in ["classic_nmt", "full"]
        self.encoder_returns = encoder_returns
        assert encoder_returns in ["EncoderOut", "dict"]

    def generate(self, sample, prefix_tokens, max_len, min_len=0):
        result = self.search(sample, prefix_tokens, max_len, min_len)
       
        n_prefix_tokens = prefix_tokens.shape[1] 

        #results = [ sorted(result, key=lambda x : x["score"] / len(x["tokens"]) ** self.length_penalty_alpha , reverse=True) ]
        results = [ sorted(result, key=lambda x : x["score"] / ( len(x["tokens"]) - n_prefix_tokens ) ** self.length_penalty_alpha , reverse=True) ]

        #for hypo in results[0]:
        #    print(hypo["score"] / len(hypo["tokens"]), hypo["tokens"])

        return results

    def search(self, sample, prefix_tokens, max_len, min_len):
        # Note: Returns a hypo where the last tokens is forced to be EOS.
        # This EOS token is included in the computations.
        # If you want to produce 6 real words, you must use max_len=6+1, to account for EOS
        assert max_len >= min_len
        with torch.no_grad():
            #shape (1 x src_length)
            orig_input_tokens = sample["net_input"]["src_tokens"]

            #print("[LL]", orig_input_tokens)
            #shape (1)
            orig_src_lengths = sample["net_input"]["src_lengths"]

            current_beam = self.beam_size

        
            n_prefix_tokens = prefix_tokens.shape[1]

            assert max_len > n_prefix_tokens     
            # We repeat the input so that it matches the "batch" of the current beam
            # In the future, it might be slightly efficient to carry out the encoding first, and then expand to the correct shape

            # (1 x src_length) -> (beam x src_length)
            input_tokens = orig_input_tokens.expand((current_beam, orig_input_tokens.shape[1]))
            # (1) -> (beam)
            src_lengths=torch.cuda.LongTensor([orig_input_tokens.shape[1]]).expand(current_beam)
            # (1 x n_prefix_tokens) -> (beam x n_prefix_tokens)
            target_tokens = prefix_tokens.expand((current_beam, prefix_tokens.shape[1]))

            # We append eos at the start
            eos_t=torch.full( (current_beam,1), self.tgt_dict.eos()).long().cuda()
            prev_output_tokens = torch.cat((eos_t, target_tokens), 1)

            encoder_output = self.model.encoder(input_tokens, src_lengths)
 
            # (beam x max_len)
            # scores holds the cumulative score of each hypothesis at each step
            scores=torch.zeros((self.beam_size, max_len), dtype=torch.float).cuda()

            finalized_sents=[]
            

            #Prefix tokens are fixed, we start decoding only for new part
            for i in range(n_prefix_tokens, max_len - 1):
                # (beam x target_length x vocab_size)
                net_output,_ = self.model.decoder(prev_output_tokens, encoder_output)
                l_probs=torch.nn.functional.log_softmax(net_output, dim=2)

                #(beam x vocab_size)
                last_l_probs = l_probs[:, -1, :]
                
                if i + 1 < min_len:
                    last_l_probs[:, self.tgt_dict.eos()] = -math.inf
                last_l_probs[:, self.tgt_dict.pad()] = -math.inf
                #Add l_probs to the scores
                last_l_probs = last_l_probs + scores[:, -1].reshape(current_beam,1)

                k = current_beam * 2

                # Combine scores into continuous array, then select top k hypotheses

                if i == n_prefix_tokens:
                    top_prediction = torch.topk(last_l_probs[0,:].view(-1),k)
                else:
                    top_prediction = torch.topk(last_l_probs.view(-1),k)
                
                # score of each of the top k hypothesis
                scores_buf = top_prediction[0]
                # index (in continuous array) of each hypothesis
                indices_buf = top_prediction[1]

                # for each hypothesis, the index of the father beam
                beams_buf = torch.div(indices_buf, torch.cuda.LongTensor([len(self.tgt_dict)]))

                # for each hypothesis, the index(in vocab) of the token
                #token_idx_buf = indices_buf.fmod(torch.cuda.LongTensor([len(self.tgt_dict)]))
                token_idx_buf = torch.fmod(indices_buf, torch.cuda.LongTensor([len(self.tgt_dict)]))

                # hypothesis with eos are considered to have finished            
                eos_mask = token_idx_buf.eq(torch.cuda.LongTensor([self.tgt_dict.eos()]))
                top_hypos = torch.arange(0, k) < current_beam
                top_hypos = top_hypos.cuda()
                valid_eos_mask = eos_mask & top_hypos
                valid_eos_indices = torch.masked_select(torch.arange(0,k).cuda(), valid_eos_mask)
                active_mask = torch.bitwise_not(eos_mask)

                # EOS hypothesis processing

                eos_beams = beams_buf[valid_eos_mask]
                eos_indices = token_idx_buf[valid_eos_mask]

                # add eos
                eos_tokens = prev_output_tokens[eos_beams]
                eos_tokens = torch.cat((eos_tokens, eos_indices.reshape(-1,1)), dim=1)
                eos_scores = scores_buf[eos_mask]
                
                for sentence, score, parent_beam  in zip(eos_tokens.tolist(), eos_scores.tolist(), eos_beams.tolist()):
                    finalized_sents.append({"tokens":sentence[1:],
                                            "score":score})

                    if self.search_mode == "classic_nmt":
                        current_beam -= 1
                    # hypos of same beam
                    same_beam = beams_buf.eq(torch.cuda.LongTensor([parent_beam]))
                    
                    # hypos with lower score
                    lower_score = scores_buf < torch.cuda.LongTensor([score])

                    to_remove = same_beam & lower_score
                    
                    if self.search_mode == "classic_nmt":
                        active_mask = active_mask & torch.bitwise_not(to_remove)

                active_hypos = torch.sum(active_mask.long())


                if current_beam == 0:
                    return finalized_sents

                # Active hypothesis processing
                new_prev_output_tokens = prev_output_tokens[beams_buf[active_mask], :]
                prev_output_tokens = torch.cat((new_prev_output_tokens,token_idx_buf[active_mask].reshape(-1,1)), dim=1)[ :min(current_beam,active_hypos) , :]
                
                #print("[LL]", i, prev_output_tokens)
                new_scores = scores[beams_buf[active_mask], :]
                scores = torch.cat((new_scores, scores_buf[active_mask].reshape(-1,1)), dim=1)[:min(current_beam, active_hypos), :]

                current_beam = min(current_beam, active_hypos)

                if self.encoder_returns == "EncoderOut":
                    encoder_output = EncoderOut(encoder_out=encoder_output.encoder_out[ : , :current_beam , : ],
                        encoder_padding_mask=encoder_output.encoder_padding_mask[:current_beam, :],
                        encoder_embedding=encoder_output.encoder_embedding[:current_beam , :, :],
                        encoder_states=None, src_tokens=None, src_lengths=None)
                elif self.encoder_returns == "dict":
                    enc_pad = None
                    if  encoder_output["encoder_padding_mask"] != None:
                        enc_pad = encoder_output["encoder_padding_mask"][:current_beam, :]
                    encoder_output = {
                                "encoder_out": encoder_output["encoder_out"][:, :current_beam, :],
                                "encoder_padding_mask": enc_pad
                    }
                else:
                    raise Exception 
                if current_beam == 0:
                    return finalized_sents
            #
            #
            if current_beam > 0:
                #For last iteration, we force to decode EOS
                net_output,_ = self.model.decoder(prev_output_tokens, encoder_output)
                l_probs=torch.nn.functional.log_softmax(net_output, dim=2)

                #(beam x vocab_size)
                last_l_probs = l_probs[:, -1, :]

                #Add l_probs to the scores
                last_l_probs = last_l_probs + scores[:, -1].reshape(current_beam,1)

                probs_eos = last_l_probs[:, self.tgt_dict.eos()]
                sentences = torch.cat((prev_output_tokens, torch.cuda.LongTensor([self.tgt_dict.eos()]).expand((current_beam,1))), dim=1)
                for sentence, score in zip(sentences.tolist(), probs_eos.tolist()):
                    finalized_sents.append({"tokens":sentence[1:],
                                                "score":score})

            return finalized_sents
