# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/core/actions/#custom-actions/


# This is a simple example for a custom action which utters "Hello World!"
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
# from data_loader import get_loader
# from build_vocab import Vocabulary
from model_fastText import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

import pandas as pd
from typing import Any, Text, Dict, List
#
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
import nltk
from gensim import models
from flask import Flask, jsonify, request, render_template
import requests
import sklearn
from collections import OrderedDict
import numpy as np
import spacy, string, nltk
from spacy.lang.en.stop_words import STOP_WORDS
import pandas as pd
import gensim
# from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')
nlp = spacy.load('en_core_web_md') 
stopwords = set(nltk.corpus.stopwords.words('english'))
punctuation = set(string.punctuation)
lemmatize = WordNetLemmatizer()
words = set(nltk.corpus.words.words())
import gpt_2_simple as gpt2

from gensim.models.wrappers import FastText

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#python -m rasa_core_sdk.endpoint --actions actions

asked_word = ""
model = ""


class TextRank4Keyword():
    """Extract keywords from text"""

    def __init__(self):
        self.d = 0.85 # damping coefficient, usually is .85
        self.min_diff = 1e-5 # convergence threshold
        self.steps = 10 # iteration steps
        self.node_weight = None # save keywords and its weight


    def set_stopwords(self, stopwords):
        """Set stop words"""
        for word in STOP_WORDS.union(set(stopwords)):
            lexeme = nlp.vocab[word]
            lexeme.is_stop = True

    def sentence_segment(self, doc, candidate_pos, lower):
        """Store those words only in cadidate_pos"""
        sentences = []
        for sent in doc.sents:
            selected_words = []
            for token in sent:
                # Store words only with cadidate POS tag
                if token.pos_ in candidate_pos and token.is_stop is False:
                    if lower is True:
                        selected_words.append(token.text.lower())
                    else:
                        selected_words.append(token.text)
            sentences.append(selected_words)
        return sentences

    def get_vocab(self, sentences):
        """Get all tokens"""
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab

    def get_token_pairs(self, window_size, sentences):
        """Build token_pairs from windows in sentences"""
        token_pairs = list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i+1, i+window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs

    def symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())

    def get_matrix(self, vocab, token_pairs):
        """Get normalized matrix"""
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1

        # Get Symmeric matrix
        g = self.symmetrize(g)

        # Normalize matrix by column
        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, where=norm!=0) # this is ignore the 0 element in norm

        return g_norm


    def get_keywords(self, number=10):
        """Print top number keywords"""
        keywords = []
        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
        for i, (key, value) in enumerate(node_weight.items()):
            keywords.append(key)
            # print(key + ' - ' + str(value))
            if i > number:
                break
        return keywords

    def analyze(self, text,
                candidate_pos=['NOUN', 'PROPN'],
                window_size=4, lower=False, stopwords=list()):
        """Main function to analyze text"""

        # Set stop words
        self.set_stopwords(stopwords)

        # Pare text by spaCy
        doc = nlp(text)

        # Filter sentences
        sentences = self.sentence_segment(doc, candidate_pos, lower) # list of list of words

        # Build vocabulary
        vocab = self.get_vocab(sentences)

        # Get token_pairs from windows
        token_pairs = self.get_token_pairs(window_size, sentences)

        # Get normalized matrix
        g = self.get_matrix(vocab, token_pairs)

        # Initionlization for weight(pagerank value)
        pr = np.array([1] * len(vocab))

        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr = (1-self.d) + self.d * np.dot(g, pr)
            if abs(previous_pr - sum(pr))  < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]

        self.node_weight = node_weight

# class Model_intialize(Action):

#     def __init__(self) -> None:
#         self.model = [1,2]

#     def name(self) -> Text:
#         return "action_model_initialize"

#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#         print("2")
        




class ActionTellDefinition(Action):

    def __init__(self) -> None:
        print(1)
        
        global model
        model = models.KeyedVectors.load_word2vec_format('PedagogicAgentBackend/embedding/wiki-news-300d-1M-subword.vec')


        #-------------------------definition model integration starts------------------------
        self.sess = gpt2.start_tf_sess()
        gpt2.load_gpt2(self.sess, run_name="run_definition")

        #--------------------------deinition model integration ends---------------------------------






        #---------------------Sri Harsha Nikhil SAI model integration starts--------------------

        # if not os.path.exists("allcontent_required_NotNull_sum_outFastText.ckpt"):
        #     os.makedirs(allcontent_required_NotNull_sum_outFastText.ckpt)

        # # Load vocabulary wrapper
        # with open("fasttext.model", 'rb') as f:
        #     self.emb_model = pickle.load(f)
        #print("1")
        # self.model = models.KeyedVectors.load_word2vec_format('crawl-300d-2M-subword/crawl-300d-2M-subword.vec')

        # self.emb_model_weights = self.emb_model.wv.syn0
        # print("2")

        # # Build data loader
        # # data_loader = get_loader(args.image_dir, args.caption_path, vocab,
        # #                             args.dictionary, args.batch_size,
        # #                             shuffle=True, num_workers=args.num_workers)
        
        # # Build the models
        # #encoder = EncoderCNN(256).to(device)
        # self.dictionary = pd.read_csv("allcontent_required_NotNull_sum_out.dict", header=0,encoding = 'unicode_escape',error_bad_lines=False)
        # self.dictionary = list(self.dictionary['keys'])
        # print("3")
        # self.decoder = DecoderRNN(256, 512, len(self.emb_model.wv.vocab), 2, self.emb_model_weights).to(device)
        # self.decoder.load_state_dict(torch.load("allcontent_required_NotNull_sum_outFastText.ckpt", map_location=device))
        # self.decoder.eval()

        #---------------------Sri Harsha Nikhil SAI model integration starts--------------------


    def name(self) -> Text:
        return "action_tell_definition"

    def keywords_in_title(self,title, cnt):
        all_keywords = []
        values = []
        item = title
        # gensim.utils.simple_preprocess(item, deacc=True)
        doc = nlp(item)
        b = []
        for tok in doc:
              if tok.is_stop != True and tok.pos_ != 'SYM' and \
                  tok.tag_ != 'PRP' and tok.tag_ != 'PRP$' and \
                    tok.tag_ != '_SP' and tok.pos_ != 'NUM' and \
                    tok.dep_ != 'aux' and tok.dep_ != 'prep' and \
                    tok.dep_ != 'det' and tok.dep_ != 'cc' and \
                    tok.lemma_ != 'frac' and len(tok) != 1 and \
                    tok.lemma_.lower() in words and \
                    tok.lemma_.lower() not in stopwords and \
                    tok.lemma_.lower() not in punctuation:
                        b.append(lemmatize.lemmatize(tok.lemma_.lower()))

        tr4w = TextRank4Keyword()
        tr4w.analyze(" ".join(b), candidate_pos = ['NOUN', 'PROPN'], window_size=4, lower=False)
        keyword = tr4w.get_keywords(cnt)
        return keyword

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        topic_name = tracker.get_slot("topic")


        count = 0
        definition = ""
        try:
            if(sklearn.metrics.pairwise.cosine_similarity(model["science"].reshape(1,-1), model[topic_name].reshape(1,-1))[0][0]>0.27):
                keywords = self.keywords_in_title("respiration plant cell oxygen water sunlight",3)
                print(keywords)
                for i in keywords:
                  if(sklearn.metrics.pairwise.cosine_similarity(model[i].reshape(1,-1), model[topic_name].reshape(1,-1))[0][0]>0.40):
                    count = 1
                if(count ==1):
                    out = gpt2.generate(self.sess,
                        run_name = "run_definition",
                        checkpoint_dir='checkpoint',
                        length=200,
                        return_as_list=True,
                        temperature=0.7,
                        prefix=topic_name,
                        truncate = "<|endoftext|>",
                        include_prefix = True, 
                        nsamples=1,
                        batch_size=1
                        )

                    definition = str(out).replace("#","is") 
                    definition += "\n\n Do you want to know more about the topic?"
                    dispatcher.utter_message(definition)    
                else:
                    out = gpt2.generate(self.sess,
                        run_name = "run_definition",
                        checkpoint_dir='checkpoint',
                        length=200,
                        return_as_list=True,
                        temperature=0.7,
                        prefix=topic_name,
                        truncate = "<|endoftext|>",
                        include_prefix = True, 
                        nsamples=1,
                        batch_size=1
                        )
                    definition = str(out).replace("#","is") 
                    definition += "\n\n This is not related to the pathway. If you want i can change your pathway"
                    dispatcher.utter_message(definition)     
            else:
                dispatcher.utter_message("Oh! That was out of my scope")
        except:
            dispatcher.utter_message("Oh! That was out of my scope")







        # if(topic_name == "respiration"):
        #   dispatcher.utter_message("a process in living organisms involving the production of energy, typically with the intake of oxygen and the release of carbon dioxide from the oxidation of complex organic substances. \n\n Do you want to know more about this?")
        #   return []
        # if(topic_name == "electron"):
        #   dispatcher.utter_message("a stable subatomic particle with a charge of negative electricity, found in all atoms and acting as the primary carrier of electricity in solids. \n\n This is not related to the pathway if you want i can change your pathway")
        #   return []
        # else:
        #   dispatcher.utter_message("Oh! That was out of my scope")
        #   return []



        # asked_word = topic_name
        # in_science_or_not = 0
        # print("cosine similarity between " + str(topic_name) + " and science is " + str(sklearn.metrics.pairwise.cosine_similarity(self.model['science'].reshape(1,-1), self.model[topic_name].reshape(1,-1))[0][0]))
        # try:
        #     if(sklearn.metrics.pairwise.cosine_similarity(self.model['science'].reshape(1,-1), self.model[topic_name].reshape(1,-1))[0][0]>0.27):
        #         in_science_or_not = 1
        #     else:
        #         dispatcher.utter_message("That was out of my scope")
        #         return []
        # except:
        #     dispatcher.utter_message("That was out of my scope")
        #     return []






        # #cleaning the title
        # test_list = ["inner ear anatomy body works organs"]
        # for i in range(len(test_list)):
        #     l = test_list[i].split(" ")
        #     new_list = []
        #     for j in l:
        #         try:
        #             score = sklearn.metrics.pairwise.cosine_similarity(self.model['science'].reshape(1,-1), self.model[j].reshape(1,-1))[0][0]
        #             if(score>0.27):
        #                 new_list.append(j)
        #         except:
        #             pass
        #     pro_str = " ".join(new_list)
        #     test_list[i] = pro_str
        # print(test_list)

        # count = 0
        # for i in test_list[0].split(" "):
        #     try:
        #         if(sklearn.metrics.pairwise.cosine_similarity(self.model[i].reshape(1,-1), self.model[topic_name].reshape(1,-1))[0][0]>0.40):
        #             print("cosine similarity between " + str(topic_name) + " and " + str(i) + " is "+ str(sklearn.metrics.pairwise.cosine_similarity(self.model[i].reshape(1,-1), self.model[topic_name].reshape(1,-1))[0][0]))
        #             count = 1
        #     except:
        #         pass
        # valid_or_not = ""
        # if(count ==1):
        #     print("valid")
        #     valid_or_not = "valid"
        # else:
        #     print("invalid")
        #     valid_or_not = "invalid"
        #Train the models
        # total_step = len(data_loader)
        # for epoch in range(5):
        # for i, (array, captions, lengths) in enumerate(data_loader):
        # if(True):
        #     data = topic_name
            
        #     array = torch.zeros((256))
        #     count = 0
        #     for val in data.split():
        #         array = torch.add(array, torch.from_numpy(self.emb_model.wv[val]))
        #         count += 1
        #     array = torch.div(array, count)
           #      # print("In sample", array)
        #     array = (array, )
        #     array = torch.stack(array, 0)
        #     array = array.to(device)
           #  # print("After", array)
           #  #captions = captions.to(device)
           #  # targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
        
           #  # Forward, backward and optimize
           #  #features = encoder(images)
        #     outputs = self.decoder.sample(array)
        
        #     count = 0
        #     sentence = ''
        #     for i in range(len(outputs)):
        #         sampled_ids = outputs[i].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
        
           #      # Convert word_ids to words
        #         sampled_caption = []
        #         for word_id in sampled_ids:
        #             count += 1
        #             word = self.emb_model.wv.index2word[word_id]
        #             if word == '<end>':
        #                 break
        #             else:
        #                 sampled_caption.append(word)
        #         sentence = sentence.join(' ')
        #         sentence = sentence.join(sampled_caption)
        #     sentence = sentence[8:]
        # if(valid_or_not=="valid"):
        #   dispatcher.utter_message("output from GPT2 will be printed here")
        # else:
        return []



class ActionTellCummulativeSummary(Action):

    def __init__(self) -> None:
        print(1)
        # URL = "http://localhost:5555/topic_data"
        # r = requests.get(url = URL) 
        # print(r.json())
        # if not os.path.exists("allcontent_required_NotNull_sum_outFastText.ckpt"):
        #     os.makedirs(allcontent_required_NotNull_sum_outFastText.ckpt)

        # # Load vocabulary wrapper
        # with open("fasttext.model", 'rb') as f:
        #     self.emb_model = pickle.load(f)
        # print("1")

        # self.emb_model_weights = self.emb_model.wv.syn0
        # print("2")

        # # Build data loader
        # # data_loader = get_loader(args.image_dir, args.caption_path, vocab,
        # #                             args.dictionary, args.batch_size,
        # #                             shuffle=True, num_workers=args.num_workers)
        
        # # Build the models
        # #encoder = EncoderCNN(256).to(device)
        # self.dictionary = pd.read_csv("allcontent_required_NotNull_sum_out.dict", header=0,encoding = 'unicode_escape',error_bad_lines=False)
        # self.dictionary = list(self.dictionary['keys'])
        # print("3")
        # self.decoder = DecoderRNN(256, 512, len(self.emb_model.wv.vocab), 2, self.emb_model_weights).to(device)
        # self.decoder.load_state_dict(torch.load("allcontent_required_NotNull_sum_outFastText.ckpt", map_location=device))
        # self.decoder.eval()


    def name(self) -> Text:
        return "action_tell_cummulative_summary"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        topic_name = "respiration plant chromozome life"
        
        # asked_word = topic_name

        # #Train the models
        # # total_step = len(data_loader)
        # # for epoch in range(5):
        # # for i, (array, captions, lengths) in enumerate(data_loader):
        # if(True):
        #     data = topic_name
            
        #     array = torch.zeros((256))
        #     count = 0
        #     for val in data.split():
        #         array = torch.add(array, torch.from_numpy(self.emb_model.wv[val]))
        #         count += 1
        #     array = torch.div(array, count)
        #         # print("In sample", array)
        #     array = (array, )
        #     array = torch.stack(array, 0)
        #     array = array.to(device)
        #     # print("After", array)
        #     #captions = captions.to(device)
        #     # targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
        
        #     # Forward, backward and optimize
        #     #features = encoder(images)
        #     outputs = self.decoder.sample(array)
        
        #     count = 0
        #     sentence = ''
        #     for i in range(len(outputs)):
        #         sampled_ids = outputs[i].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
        
        #         # Convert word_ids to words
        #         sampled_caption = []
        #         for word_id in sampled_ids:
        #             count += 1
        #             word = self.emb_model.wv.index2word[word_id]
        #             if word == '<end>':
        #                 break
        #             else:
        #                 sampled_caption.append(word)
        #         sentence = sentence.join(' ')
        #         sentence = sentence.join(sampled_caption)
        #     sentence = sentence[8:]
        dispatcher.utter_message(topic_name)
        return []



class ActionGiveResource(Action):
    def __init__(self) -> None:
        x = 1
        # self.model = models.KeyedVectors.load_word2vec_format('wiki-news-300d-1M-subword.vec')
        self.embeddings = np.load("PedagogicAgentBackend/data/embeddings.npy")
        self.words = open("PedagogicAgentBackend/data/words.txt", "r").read().split()
        self.URL = open("PedagogicAgentBackend/data/URL.txt", "r").read().split()
        self.nbrs = NearestNeighbors(n_neighbors=1).fit(self.embeddings)


    def name(self) -> Text:
        return "action_give_resource"
#     def get_nearest_word(word):
#         try:
#             emb = model[word]
#         except:
#     # emb = np.random.rand(300)
#             emb = np.zeros(300)

#         emb = np.array(emb).reshape(1,-1)
#         distance, index = nbrs.kneighbors(emb)
#         word = words[index[0][0]]
        

#     def get_nearest_url(word):
#         word = get_nearest_word(word)
#         idx = word_idx[word]
#         return URL[idx]

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        word = tracker.get_slot("topic")
        print(word)
        try:
            emb = model[word]
        except:
    # emb = np.random.rand(300)
            emb = np.zeros(300)

        emb = np.array(emb).reshape(1,-1)
        distance, index = self.nbrs.kneighbors(emb)
        nearest_word = self.words[index[0][0]]
        url = self.URL[index[0][0]]
        print(nearest_word)
        # idx = word_idx[nearest_word]
        dispatcher.utter_message(url)
        return []
       


app = Flask(__name__)

@app.route('/hello', methods=['GET',])
def hello():
    message = {'greeting':'Hello from Flask!'}
    return jsonify(message)








