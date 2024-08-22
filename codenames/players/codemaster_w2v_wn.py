import scipy.spatial.distance
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import itertools

from players.codemaster import Codemaster


class AICodemaster(Codemaster):

    def __init__(self, brown_ic=None, glove_vecs=None, word_vectors=None):
        super().__init__()
        self.brown_ic = brown_ic
        self.glove_vecs = glove_vecs
        self.word_vectors = word_vectors
        self.wordnet_lemmatizer = WordNetLemmatizer()
        self.lancaster_stemmer = LancasterStemmer()
        self.cm_wordlist = []
        with open('players/cm_wordlist.txt') as infile:
            for line in infile:
                self.cm_wordlist.append(line.rstrip())
        
        # calculate synsets for wordnet calculations
        self.syns = []
        for word in self.cm_wordlist:
            for synset_in_cmwordlist in wordnet.synsets(word):
                self.syns.append(synset_in_cmwordlist)

        self.bad_word_dists = None
        self.red_word_dists = None

    def set_game_state(self, words, maps):
        self.words = words
        self.maps = maps

    # gets clue using both w2v and wn distance metrics
    def get_clue(self):
        cos_dist = scipy.spatial.distance.cosine
        red_words = []
        bad_words = []

        # Categorize red and bad (all other) words
        for i in range(25):
            if self.words[i][0] == '*':
                continue
            elif self.maps[i] == "Assassin" or self.maps[i] == "Blue" or self.maps[i] == "Civilian":
                bad_words.append(self.words[i].lower())
            else:
                red_words.append(self.words[i].lower())

        print("RED:\t", red_words)

        
        best_hint = None
        best_similarity = 0.0
        for word in self.cm_wordlist:
            #print(word)
            if word in red_words or word in bad_words:
                continue
            if not self.arr_not_in_word(word, red_words + bad_words):
                        continue

            # wordnet similarity score
            wn_red_word, wn_sim, wn_sims = self.get_wn_similarity(word, red_words, bad_words)

            # word2vec similarity score
            w2v_red_word, w2v_sim, w2v_sims = self.get_w2v_similarity(word, red_words, bad_words)

            # combine them (HERE WE SHOULD TEST DIFFERENT WEIGHTS)
            total_score = (wn_sim + w2v_sim) / 2.0


            if total_score > best_similarity:
                best_hint = word
                best_similarity = total_score
                w2v_word = w2v_red_word
                wn_word = wn_red_word
                wn_dict = wn_sims
                w2v_dict = w2v_sims

                print("best sim: ", best_similarity, "\nbest hint: ", best_hint)

        
        print("Wordnet word: ", wn_word , "\nWord2Vec word: ", w2v_word, "\nwn_sims", wn_dict, "\nw2v_sims", w2v_dict, "\nHint: ", best_hint)
        
        # IMPLEMENT NEW WAY OF PICKING NUMBER USING 'wn_dict' AND 'w2v_dict'
        
        # if the two algorithms pick the same red word (for max similarity), return 1. otherwise return 2
        if wn_word == w2v_word:
            return (best_hint, 1)
        else:
            return (best_hint,2)
        
        # could also just return (best_hint,1)
        #return (best_hint, 1)
    
    # gets w2v similarity scores using cosine distance (returns a dict: word -> similarity)
    def get_w2v_similarity(self, word, red_words, bad_words):
        cos_dist = scipy.spatial.distance.cosine
        all_vectors = (self.word_vectors,)
        
        best_sim = 0.0
        similar_word = None
        word_sims = {}
        for red_word in red_words:
            red_sim = 1 - cos_dist(self.concatenate(word, all_vectors), self.concatenate(red_word, all_vectors))
            word_sims[red_word] = red_sim
            #print("best sim: ", best_sim, "\nsim: ", red_sim)
            if red_sim > best_sim:
                #print("word: ", word, "\nred word: ", red_word, "\nsim: ", red_sim)
                best_sim = red_sim
                similar_word = red_word


        return similar_word, best_sim, word_sims
    
    # gets wn similarity using lin similarity (returns a dict: word -> similarity)
    # we can test different similarity metrics, but lin similarity is already implemented in the codebase
    def get_wn_similarity(self, word, red_words, bad_words):
        best_sim = 0.0
        similar_word = None
        word_sims = {}

        # Get synsets for the given word (the potential clue)
        word_synsets = wordnet.synsets(word)

        for red_word in red_words:
            red_word_synsets = wordnet.synsets(red_word)
            for word_synset in word_synsets:
                for red_synset in red_word_synsets:
                    try:
                        score = word_synset.lin_similarity(red_synset, self.brown_ic)
                        word_sims[red_word] = score
                        if score and score > best_sim:
                            best_sim = score
                            similar_word = red_word
                    except:
                        continue
        

        return similar_word, best_sim, word_sims

    
    def arr_not_in_word(self, word, arr):
        if word in arr:
            return False
        lemm = self.wordnet_lemmatizer.lemmatize(word)
        lancas = self.lancaster_stemmer.stem(word)
        for i in arr:
            if i == lemm or i == lancas:
                return False
            if i.find(word) != -1:
                return False
            if word.find(i) != -1:
                return False
        return True
    

    # This method concatenates wordvectors from muliple sources (ie word2vec, gloVe) to create 1 vector
    def concatenate(self, word, wordvecs):
        concatenated = wordvecs[0][word]
        for vec in wordvecs[1:]:
            concatenated = np.hstack((concatenated, vec[word]))
        return concatenated



