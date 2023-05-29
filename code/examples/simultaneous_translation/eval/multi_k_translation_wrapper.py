from fairseq.registry import REGISTRIES
from fairseq import checkpoint_utils, utils, tasks
from fairseq import search, sequence_generator, incremental_simultaneous_beam_search, incremental_simultaneous_beam_search_no_forced_eos
import os
import json
import torch
import math
import re
import copy
import argparse
import inspect
import time
from nltk.util import ngrams

class SentencePieceModelWordSplitterIdInput(object):
    def __init__(self, model_path, debug=False):
        import sentencepiece as spm
        self.model = spm.SentencePieceProcessor()
        self.model.Load(model_path)
        self.debug = debug

    def split(self, string):
        #Returns a list of strings/pieces
        if string == '</s>':
            return ['</s>']
        else:
            return [str(x) for x in self.model.EncodeAsIds(string)]

    def is_end_word(self, token):
        piece = self.model.IdToPiece(int(token))

        #Assume whitespace as suffix
        return piece[-1] == '\u2581'

    def merge(self, list_of_string):

        pieces = []

        for ind in list_of_string:
            if ind == self.model['<unk>']:
                pieces.append('<unk>▁')
            else:
                pieces.append(self.model.IdToPiece(int(ind)))

        return "".join(pieces).replace("▁"," ")



class MultikTranslationWrapper:

    def load_dictionary(self, task):
        self.dict = {}
        self.dict["tgt"] = task.target_dictionary
        self.dict["src"] = task.source_dictionary

    def load_model(self, args):
        args.user_dir = os.path.join(os.path.dirname(__file__), '..')
        utils.import_user_module(args)
        filename = args.model_path
        if not os.path.exists(filename):
            raise IOError("Model file not found: {}".format(filename))

        state = checkpoint_utils.load_checkpoint_to_cpu(filename)

        saved_args = state["args"]
        saved_args.data = args.data_bin

        task = tasks.setup_task(saved_args)

        # build model for ensemble
        self.model = task.build_model(saved_args)
        self.model.load_state_dict(state["model"], strict=True)

        use_cuda = torch.cuda.is_available() #and not args.cpu
        if use_cuda:
            self.model.cuda()
            print("Using CUDA")

        # Set dictionary
        self.load_dictionary(task)
        self.model.eval()

    def __init__(self):
        args = self.get_args()
        self.args = args 
        self.load_model(args)
        search_strategy = search.BeamSearch(self.dict["tgt"])

        self.efficient_generator = incremental_simultaneous_beam_search.IncrementalSimultaneousBeamSearch(
            self.model,
            self.dict["tgt"],
            beam_size=getattr(args, "beam"),
            length_penalty_alpha=getattr(args, "length_penalty_alpha"),
            search_strategy=search_strategy,
            encoder_returns="dict")

        self.efficient_generator_no_forced_eos = incremental_simultaneous_beam_search_no_forced_eos.IncrementalSimultaneousBeamSearchNoForcedEOS(
            self.model,
            self.dict["tgt"],
            beam_size=getattr(args, "beam"),
            length_penalty_alpha=getattr(args, "length_penalty_alpha"),
            search_strategy=search_strategy,
            encoder_returns="dict")

        self.word_splitter = {}

        self.word_splitter["src"] = SentencePieceModelWordSplitterIdInput(
                getattr(args, f"src_splitter_path"), self.args.debug
            )
        self.word_splitter["tgt"] = SentencePieceModelWordSplitterIdInput(
                getattr(args, f"tgt_splitter_path"), self.args.debug
            )

        self.special_tokens = {
            "src_doc":"[DOC]",
            "tgt_doc":"[DOC]",   
            "src_cont":"[CONT]",   
            "tgt_cont":"[CONT]",   
        }
    def get_special_tokens(self, special_tokens_file):
        self.special_tokens = {}
        with open(special_tokens_file) as f:
            for line in f:
                ks = line.strip().split()
                assert len(ks) == 2
                key = ks[0]
                value = ks[1]
                self.special_tokens[key]=value

    def get_args(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--model-path', type=str, help='Path to the model to be used for inference')
        parser.add_argument('--data-bin', type=str, help='Path to the folder containing the model dictionary')
        parser.add_argument('--src-file', type=str, help='Path to the input file')
        parser.add_argument('--output', type=str, help='Path to the output folder')
        parser.add_argument("--max-history-size-src", type=int,
                            help="The maximum amount of words kept in history")
        parser.add_argument("--max-history-size-tgt", type=int,
                            help="The maximum amount of words kept in history")
        parser.add_argument("--history-mode", choices=['truncate','strict'], default='strict',
                            help="Strict=Discard if length exceeded, Truncate=Truncate at the word level")
        parser.add_argument("--user-dir", type=str, default="example/simultaneous_translation",
                            help="User directory for simultaneous translation")
        parser.add_argument("--src-splitter-path", type=str, default=None,
                            help="Subword splitter model path for source text")
        parser.add_argument("--tgt-splitter-path", type=str, default=None,
                            help="Subword splitter model path for source text")

        parser.add_argument("--simul-waitk", type=int, default=3,
                            help="Wait-k used for simultaneous evaluation")

        parser.add_argument("--catchup", type=float, default=1.0,
                            help="Usually set to avg. tgt_len / src_len. We write 'catchup' target tokens for every source token read.")
        
        parser.add_argument("--always-finish-read", action='store_true',
                            help="Force decoding")

        parser.add_argument("--debug-times", action='store_true')
        parser.add_argument("--debug", action='store_true', help="General purpouse debug")

        parser.add_argument("--reset-history-if-stuck", action='store_true',
                            help="Resets history if detects model has become stuck")


        parser.add_argument("--max-len", type=int, default=130)

        # beam search arguments
        parser.add_argument("--beam",type=int, default=1, help="Beam-search parameter")
        parser.add_argument("--max-len-a",type=int, default=2, help="Generate sequences of max length [catchup]*x + a, x being source length")
        parser.add_argument("--length-penalty-alpha",type=float, default=1.0, help="score(Y,X)=lp(P(Y/X)/ len(Y)^alpha ")

        args, _ = parser.parse_known_args()

        for registry_name, REGISTRY in REGISTRIES.items():
            choice = getattr(args, registry_name, None)
            if choice is not None:
                cls = REGISTRY["registry"][choice]
                if hasattr(cls, "add_args"):
                    cls.add_args(parser)

        args = parser.parse_args()

        return args

    def compute_length_buffer(self,buf):
        length=0
        for s in buf:
            length+= len(s)
        return length
      
    def prepare_buffers(self, s_buf, t_buf):
        # Source/target buffer is a list of lists.
        # Each element represents a sentence.
        # Each sub-element represents a unit that will be "sent/received" alltogether.
        # In practice, each sub-element is a (detokenized) word, which is then processed by the splitter.
        length_limit_src = self.args.max_history_size_src
        length_limit_tgt = self.args.max_history_size_tgt

        if self.args.history_mode == "words":
            len_s = self.compute_length_buffer(s_buf)
            while len_s > length_limit_src:
                if len(s_buf) > 1 and len_s - len(s_buf[0]) >= length_limit_src:
                    len_s -= len(s_buf[0])
                    s_buf.pop(0)
                elif len(s_buf) > 1:
                    len_current = len(s_buf[0])
                    keep = length_limit_src - (len_s - len_current)
                    list_to_keep = s_buf[0][-keep:]
                    assert len(list_to_keep) + (len_s - len_current) == length_limit_src
                    s_buf[0] = list_to_keep
                    break
                else:
                    words = s_buf[0]
                    list_to_keep = words[-length_limit_src:]
                    s_buf[0] = list_to_keep
                    break

            len_t = self.compute_length_buffer(t_buf)
            while len_t > length_limit_tgt:
                if len(t_buf) > 1 and len_t - len(t_buf[0]) >= length_limit_tgt:
                    len_t -= len(t_buf[0])
                    t_buf.pop(0)
                elif len(t_buf) > 1:
                    len_current = len(t_buf[0])
                    keep = length_limit_tgt - (len_t - len_current)
                    list_to_keep = t_buf[0][-keep:]
                    assert len(list_to_keep) + (len_t - len_current) == length_limit_tgt
                    t_buf[0] = list_to_keep
                    break
                else:
                    words = t_buf[0]
                    list_to_keep = words[-length_limit_tgt:]
                    t_buf[0] = list_to_keep
                    break

            return s_buf, t_buf

        elif self.args.history_mode == "strict":
            while self.compute_length_buffer(s_buf) > length_limit_src or self.compute_length_buffer(t_buf) > length_limit_tgt:
                if len(s_buf) > 1:
                    s_buf.pop(0)
                    t_buf.pop(0)
                else:
                    s_buf = [[]]
                    t_buf = [[]]

            return s_buf, t_buf
        else:
            raise NotImplementedError

    def make_history_strings(self, src_buffer, tgt_buffer, is_start):

        src_string = ""
        tgt_string = ""
        

        if self.args.max_history_size_src > 0:
            if is_start:
                src_string=self.special_tokens["src_doc"]
            else:
                src_string=self.special_tokens["src_cont"]

            for sent in src_buffer:
                src_string += " "
                src_string += " ".join(sent)

        if self.args.max_history_size_tgt > 0:
            if is_start:
                tgt_string=self.special_tokens["tgt_doc"]
            else:
                tgt_string=self.special_tokens["tgt_cont"]

            for sent in tgt_buffer:
                tgt_string += " "
                tgt_string += " ".join(sent)
        
        return src_string, tgt_string


    def get_max_ngram_repeat(self,sequence, up_to_order=6):
        """ Receive an iterable. Return the maximum number of times an element is repeated"""

        assert up_to_order >= 2

        if len(sequence) == 0:
            return 0

        else:
            max_count = 0
            for n in range(2, up_to_order+1):
                elements = list(ngrams(sequence, n))
                if len(elements) > 1:
                    last_ngram = elements[0]
                    count = 1
                    for elem in elements[1:]:
                        if elem == last_ngram:
                            count +=1
                            if count > max_count:
                                max_count = count
                        else:
                            count = 1
                            last_ngram = elem

            return max_count



    def reset_history(self,last_source_sentence, last_target_sentence):
        """Return value: bool. Checks to see if it is needed to reset history"""
        MAX_NGRAM_REPEAT = 5

        MIN_SENTENCE_LENGTH_FOR_FILTER = 10
        MIN_CHARACTER_LENGTH_FOR_FILTER = 30
        MIN_LEN_RATIO = 0.4
        MAX_LEN_RATIO = 3
    

        if self.get_max_ngram_repeat(last_target_sentence) > MAX_NGRAM_REPEAT or self.get_max_ngram_repeat(list("".join(last_target_sentence))) > MAX_NGRAM_REPEAT:
            return True
        elif len(last_source_sentence) > MIN_SENTENCE_LENGTH_FOR_FILTER and len(last_target_sentence) > MIN_SENTENCE_LENGTH_FOR_FILTER and len(last_source_sentence)/len(last_target_sentence) < MIN_LEN_RATIO:
            return True
        elif len(last_source_sentence) > MIN_SENTENCE_LENGTH_FOR_FILTER and len(last_target_sentence) > MIN_SENTENCE_LENGTH_FOR_FILTER and len(last_source_sentence)/len(last_target_sentence) > MAX_LEN_RATIO:
            return True
        elif len(list("".join(last_source_sentence))) > MIN_CHARACTER_LENGTH_FOR_FILTER and len(list("".join(last_target_sentence))) > MIN_CHARACTER_LENGTH_FOR_FILTER  and len(list("".join(last_target_sentence)))/len(list("".join(last_source_sentence))) < MIN_LEN_RATIO:
            return True
        elif len(list("".join(last_source_sentence))) > MIN_CHARACTER_LENGTH_FOR_FILTER and len(list("".join(last_target_sentence))) > MIN_CHARACTER_LENGTH_FOR_FILTER and len(list("".join(last_target_sentence)))/len(list("".join(last_source_sentence))) > MAX_LEN_RATIO:
            return True
        else:
            return False

    def get_prediction(self, states):
        if self.args.debug_times:
            start_time_debug = time.time()
        #Assume GPU
        tensor = torch.cuda.LongTensor
        
        src_indices = tensor([states["indices"]["src"]])

        tgt_indices = tensor([[self.model.decoder.dictionary.eos()] + states["indices"]["tgt"]])

        encoder_out_dict = self.model.encoder(src_indices, None)

        x,_=self.model.decoder.forward(prev_output_tokens=tgt_indices,encoder_out=encoder_out_dict)

        lprobs = self.model.get_normalized_probs(
            [x[:, -1:]],
            log_probs=True
        )

        index = lprobs.argmax(dim=-1)

        token = self.model.decoder.dictionary[index]

        if self.args.debug_times:
            end_time_debug = time.time()
            print("[DDTT]", end_time_debug - start_time_debug)
        return token, index[0, 0].item()

    def get_prediction_beam(self, states):
        if self.args.debug_times:
            start_time_debug = time.time()

        tensor = torch.cuda.LongTensor
        src_indices = tensor([states["indices"]["src"]])
        prefix_tokens = tensor([states["indices"]["tgt"]])

        sample = {'net_input': {
                    'src_tokens': src_indices,
                    'src_lengths': tensor([src_indices.shape[1]]),
                }}

        max_generation_length=len(states["indices"]["tgt"]) + 1 + self.args.max_len_a
        hypos = self.efficient_generator_no_forced_eos.generate(sample, prefix_tokens, min(max_generation_length, self.args.max_len + 2))
        write_buffer = hypos[0][0]["tokens"][len(states["indices"]["tgt"]):]
        next_token, index = self.model.decoder.dictionary[write_buffer[0]], write_buffer[0]

        if self.args.debug_times:
            end_time_debug = time.time()
            print("[DDTT]", end_time_debug - start_time_debug)
 
        return next_token, index

    def init_states(self):
        states = {}
        states["indices"] = {}
        states["segments"] = {}
        states["finished_read"] = False
        states["finished_write"] = False
        states["indices"]["src"] = []
        states["indices"]["tgt"] = []
        states["segments"]["src"] = []
        states["segments"]["tgt"] = []
        states["src_context_len"] = 0
        states["tgt_context_len"] = 0
        states["actions"] = []
        states["valid_actions"] = []
        #Indicates the number of words that have been read
        states["read_words"] = 0

        return states

    def decode(self):

        with open(self.args.src_file) as f:
            src_sentences = [r.strip() for r in f]
       
        translated_sentences = []
        valid_actions = []
 
        #Remove last token to exclude [BRK] from history
        source_history = [ src_sentences[0].split(" ")[:-1] ]

        init_src_history = source_history[0]

        src_context=""
        tgt_context=""
        
        
        if self.args.max_history_size_src > 0:
            src_context = self.special_tokens["src_doc"]

        if self.args.max_history_size_tgt > 0:
            tgt_context = self.special_tokens["tgt_doc"]

        decoded_sentence, actions = self._decode_one(src_sentences[0], src_context, tgt_context)

        print(0, src_sentences[0], decoded_sentence, actions)

        while len(decoded_sentence) > 0 and decoded_sentence[-1] in ["[BRK]", "</s>"]:
                decoded_sentence.pop(-1)

        translated_sentences.append(decoded_sentence)
        valid_actions.append(actions)

        target_history = [ decoded_sentence ]

        for i in range(1, len(src_sentences)):
            new_src_history, new_tgt_history = self.prepare_buffers(source_history, target_history)
            source_history = new_src_history
            target_history = new_tgt_history
            try:
                start_doc = source_history[0] == init_src_history
            except:
                start_doc = False

            src_context, tgt_context = self.make_history_strings(source_history, target_history, start_doc)

            source_history.append(src_sentences[i].split(" ")[:-1])
            decoded_sentence, actions = self._decode_one(src_sentences[i], src_context, tgt_context)

            print(i, src_sentences[i], decoded_sentence, actions)

            while len(decoded_sentence) > 0 and decoded_sentence[-1] in ["[BRK]", "</s>"]:
                    decoded_sentence.pop(-1)

            translated_sentences.append(decoded_sentence)
            valid_actions.append(actions)

            target_history.append(decoded_sentence)

            if self.args.reset_history_if_stuck and self.reset_history(source_history[-1],target_history[-1]):
                source_history = []
                target_history = []            
                       

        os.makedirs(self.args.output, exist_ok=True)

        with open(self.args.output + "/sentence_valid_delays", "w") as svd, open(self.args.output + "/text", "w") as trans_file:
            for orig_sentence, translated_sentence, sentence_actions in zip(src_sentences, translated_sentences, valid_actions):
                counter=0
                delays_a=[]
                for action in sentence_actions:
                    if action[0] == "R":
                        counter+=1
                    else:
                        delays_a.append(counter)
                
                #Make sure that the whole sentence is read

                if self.args.max_history_size_src > 0:
                    orig_eos=orig_sentence.strip().split(" ")[:-2]
                else:
                    orig_eos = orig_sentence.strip().split(" ")
                counter= len(orig_eos) + 1
                
                trans_file.write(" ".join(translated_sentence)+"\n")

                svd.write(
                    json.dumps(
                    {
                        "src_len": counter,
                        "delays": delays_a
                    }
                ) + "\n"
                ) 

    def _decode_one(self,sentence, src_history_string, tgt_history_string):
        with torch.no_grad():
            self.model.eval()

            states = self.init_states()

            splitted_sentence = sentence.split(" ")

            if len(src_history_string) > 0:
                states["segments"]["src"] = src_history_string.split(" ")
                pieces = self.word_splitter["src"].split(src_history_string)
                states["indices"]["src"] = self.dict["src"].encode_line(
                    pieces,
                    line_tokenizer=lambda x: x,
                    add_if_not_exist=False,
                    append_eos=False
                ).tolist()

                states["src_context_len"] = len(states["segments"]["src"])

            if len(tgt_history_string) > 0:
                states["segments"]["tgt"] = tgt_history_string.split(" ")
                pieces = self.word_splitter["tgt"].split(tgt_history_string)
                states["indices"]["tgt"] = self.dict["tgt"].encode_line(
                    pieces,
                    line_tokenizer=lambda x: x,
                    add_if_not_exist=False,
                    append_eos=False
                ).tolist()

                states["tgt_context_len"] = len(states["segments"]["tgt"])

            print("init states", states)

            if self.args.debug:
                print("[DD] splitted_sentence", splitted_sentence)
            #Always read first
            states, _ , _ = self.read_action(states, splitted_sentence)

            while not states["finished_write"]:
                gamma = self.args.catchup
                # The next token to be written
                t = len(states["segments"]["tgt"]) - states["tgt_context_len"] + 1
                g_t = math.ceil(self.args.simul_waitk + (t - 1) / gamma)


                if states["finished_read"]:
                    if self.args.beam > 1:
                        tensor = torch.cuda.LongTensor
                        src_indices = tensor([states["indices"]["src"]])
                        prefix_tokens = tensor([states["indices"]["tgt"]])

                        sample = {'net_input': {
                                    'src_tokens': src_indices,
                                    'src_lengths': tensor([src_indices.shape[1]]),
                                }}

                        hypos = self.efficient_generator.generate(sample, prefix_tokens, self.args.max_len + 2)
                        write_buffer = hypos[0][0]["tokens"][len(states["indices"]["tgt"]):]
    
                        # Pop eos, will be added later                                               
                        write_buffer.pop(-1)

                        #Otherwise, we get empty string when merging
                        if len(write_buffer) > 0:
                            write_buffer_tokens = [ self.model.decoder.dictionary[x] for x in write_buffer]
                            new_string = self.word_splitter["tgt"].merge(write_buffer_tokens)
                            for segment in new_string.strip().split(" "):
                                states["segments"]["tgt"].append(segment)
                                states["actions"].append( ("W", segment))
                                if segment not in ["[SEP]", "[BRK]"]:
                                    states["valid_actions"].append( ("W", segment))

                        #Lastly, add EOS
                        states["finished_write"] = True
                        states["segments"]["tgt"].append("</s>")
                        states["actions"].append(  ("W", "</s>"))
                        states["valid_actions"].append(  ("W", "</s>"))
                        break

                    else:
                        decision = 1
                elif (len(states["segments"]["src"]) - states["src_context_len"]) < g_t:
                    decision = 0

                else:
                    decision = 1

                if decision == 1:
                    new_states, new_indices, new_segments = self.write_action(states)
                    
                    if self.args.always_finish_read and new_segments[-1] in ["</s>", "[SEP]", "[BRK]"] and not states["finished_read"]:
                        new_states, new_indices, new_segments = self.read_action(states, splitted_sentence)
                        if self.args.debug:
                            print("[DD] R", new_indices, new_segments, new_states)
                    elif self.args.debug:
                        print("[DD] W", new_indices, new_segments, new_states)
                    states = new_states

                else:
                    states, new_indices, new_segments = self.read_action(states, splitted_sentence)
                    if self.args.debug:
                        print("[DD] R", new_indices, new_segments, states)

            #Si se ha tratado con spm, los segments llevan un espacio al final que hay que eliminar
            to_retu = states["segments"]["tgt"][states["tgt_context_len"]:]
            processed_retu = [ x.strip() for x in to_retu ]

            print("end states", states)

            return processed_retu, states["valid_actions"]

    def write_action(self, states):
        tokens = []
        indices = []

        tmp_states = copy.deepcopy(states)

        if self.args.beam > 1:
            token, index = self.get_prediction_beam(tmp_states)
        else:
            token, index = self.get_prediction(tmp_states)

        torch.cuda.empty_cache()

        tokens.append(token)
        indices.append(index)
        tmp_states["indices"]["tgt"].append(index)

        while not tokens[-1] == "</s>" and not self.word_splitter["tgt"].is_end_word(tokens[-1]) and len(tmp_states["indices"]["tgt"]) < self.args.max_len:
            if self.args.beam > 1:
                token, index = self.get_prediction_beam(tmp_states)
            else:
                token, index = self.get_prediction(tmp_states)

            torch.cuda.empty_cache()
            tokens.append(token)
            indices.append(index)
            tmp_states["indices"]["tgt"].append(index)


        if len(tmp_states["indices"]["tgt"]) == self.args.max_len and not tokens[-1] == "</s>" and not self.word_splitter["tgt"].is_end_word(tokens[-1]):
            ind_space = self.word_splitter["tgt"].model.PieceToId("▁")
            tokens.append(str(ind_space))
            index = self.model.decoder.dictionary.index(str(ind_space))
            tmp_states["indices"]["tgt"].append(index)

        add_eos = tokens[-1] == "</s>"

        if add_eos:
            tokens.pop(-1)
        new_segments = []

        if len(tokens) > 0:
            new_segment = self.word_splitter["tgt"].merge(tokens)
            tmp_states["segments"]["tgt"].append(new_segment)
            if new_segment.strip() not in ["[SEP]", "[BRK]"]:
                tmp_states["valid_actions"].append(  ("W", new_segment))
            tmp_states["actions"].append(  ("W", new_segment))
            new_segments.append(new_segment)

        if add_eos:
            tmp_states["finished_write"] = True
            new_segments.append("</s>")
            tmp_states["segments"]["tgt"].append("</s>")            
            tmp_states["actions"].append(  ("W", "</s>"))
            tmp_states["valid_actions"].append(  ("W", "</s>"))
        
        if len(tmp_states["indices"]["tgt"]) >= self.args.max_len and not tmp_states["finished_write"]:
            tmp_states["segments"]["tgt"].append("</s>")
            tmp_states["finished_write"] = True
            tmp_states["actions"].append(  ("W", "</s>"))
            tmp_states["valid_actions"].append(  ("W", "</s>"))
 
        return tmp_states, indices, new_segments

    def read_action(self, states, splitted_sentence):
        if states["read_words"] < len(splitted_sentence):
            new_segment = splitted_sentence[states["read_words"]]
            states["read_words"] += 1
            new_indices = self.dict["src"].encode_line(
                self.word_splitter["src"].split(new_segment),
                line_tokenizer=lambda x: x,
                add_if_not_exist=False,
                append_eos=False
            ).tolist()

        elif states["read_words"] == len(splitted_sentence):
            states["finished_read"] = True
            new_segment = "</s>"
            new_indices = self.dict["src"].encode_line(
                self.word_splitter["src"].split(new_segment),
                line_tokenizer=lambda x: x,
                add_if_not_exist=False,
                append_eos=False
            ).tolist()

 
        else:
            raise Exception

        states["indices"]["src"].extend(new_indices)

        if len(states["indices"]["src"]) > self.args.max_len:
            states["indices"]["src"].extend(self.dict["src"].encode_line(
                self.word_splitter["src"].split("</s>"),
                line_tokenizer=lambda x: x,
                add_if_not_exist=False,
                append_eos=False
            ).tolist())
            states["finished_read"] = True

        states["actions"].append( ("R",new_segment) )
        if new_segment not in ["[BRK]", "[SEP]"]:
            states["valid_actions"].append( ("R",new_segment) )
        states["segments"]["src"].append(new_segment)

        return states, new_indices, new_segment

if __name__ == "__main__":
    wrapper = MultikTranslationWrapper() 
    wrapper.decode()
            
