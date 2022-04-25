# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/core/actions/#custom-actions/
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import pandas as pd
from typing import Any, Text, Dict, List
import spacy
import sklearn
import numpy as np
from numpy.linalg import norm

class ActionTellDefinition(Action):

    def __init__(self) -> None:
        
        model = None

        #-------------------------definition model integration starts------------------------

        #--------------------------deinition model integration ends---------------------------------

    def name(self) -> Text:
        return "action_tell_definition"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        topic_name = tracker.get_slot("topic")
        print(topic_name)

        nlp = spacy.load('en_core_web_md')
        
        cosine = lambda A,B: np.sum(A*B, axis=1)/(norm(A, axis=1)*norm(B, axis=1))

        count = 0
        definition = ""
        
        print(cosine(nlp("science").vector.reshape(1,-1), nlp(topic_name).vector.reshape(1,-1))[0])

        try:
            if(cosine(nlp("science").vector.reshape(1,-1), nlp(topic_name).vector.reshape(1,-1))[0]>0.27):
                keywords = ["respiration plant cell oxygen water sunlight"]
                print(keywords)
                for i in keywords:
                  if(cosine(nlp(i).vector.reshape(1,-1), nlp(topic_name).vector.reshape(1,-1))[0]>0.40):
                    count = 1
                if(count ==1):
                    out = "{} then there is defination of the word here".format(topic_name)

                    definition = str(out).replace("#","is") 
                    definition += "\n\n Do you want to know more about the topic?"
                    dispatcher.utter_message(definition)    
                else:
                    out = "{} then there is defination of the word here".format(topic_name)

                    definition = str(out).replace("#","is") 
                    definition += "\n\n This is not related to the pathway. If you want i can change your pathway"
                    dispatcher.utter_message(definition)     
            else:
                dispatcher.utter_message("Oh! That was out of my scope")
        except:
            dispatcher.utter_message("Oh! That was out of my scope")


class ActionGiveResource(Action):
    def __init__(self) -> None:
        x = 1
        # self.model = models.KeyedVectors.load_word2vec_format('wiki-news-300d-1M-subword.vec')
        # self.embeddings = np.load("PedagogicAgentBackend/data/embeddings.npy")
        # self.words = open("PedagogicAgentBackend/data/words.txt", "r").read().split()
        # self.URL = open("PedagogicAgentBackend/data/URL.txt", "r").read().split()
        # self.nbrs = NearestNeighbors(n_neighbors=1).fit(self.embeddings)


    def name(self) -> Text:
        return "action_give_resource"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        word = tracker.get_slot("topic")
        
        print(word)
    #     try:
    #         emb = model[word]
    #     except:
    # # emb = np.random.rand(300)
    #         emb = np.zeros(300)

        # emb = np.array(emb).reshape(1,-1)
        # distance, index = self.nbrs.kneighbors(emb)
        # nearest_word = self.words[index[0][0]]
        # url = self.URL[index[0][0]]
        # print(nearest_word)
        # idx = word_idx[nearest_word]
        dispatcher.utter_message("wikipedia.org")
        return []
       