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
        best_red_words = []


        for word in self.cm_wordlist:
            if word in red_words or word in bad_words:
                continue

            # wordnet similarity score
            wn_red_word, wn_sim = self.get_wn_similarity(word, red_words, bad_words)

            # word2vec similarity score
            w2v_red_word, w2v_sim = self.get_w2v_similarity(word, red_words, bad_words)

            # combine them (HERE WE SHOULD TEST DIFFERENT WEIGHTS)
            total_score = (wn_sim + w2v_sim) / 2.0

            similar_red_words = [
            rw for rw in red_words
            if (self.get_wn_similarity(word, [rw], bad_words)[1] > 0.4 or
                self.get_w2v_similarity(word, [rw], bad_words)[1] > 0.4)
        ]

            if total_score > best_similarity and similar_red_words:
                best_hint = word
                best_similarity = total_score
                best_red_words = similar_red_words

            

        # if the two algorithms pick the same red word (for max similarity), return 1. otherwise return 2
        if len(best_red_words) > 1:
            return (best_hint, len(best_red_words))
        else:
            return (best_hint, 1)

        # could also just return (best_hint,1)
        #return (best_hint, 1)
    
    # gets w2v similarity scores using cosine distance (returns a dict: word -> similarity)
    def get_w2v_similarity(self, word, red_words, bad_words):
        cos_dist = scipy.spatial.distance.cosine
        all_vectors = (self.word_vectors,)
        
        best_sim = 0.0
        similar_word = None
        for red_word in red_words:
            red_sim = 1 - cos_dist(self.concatenate(word, all_vectors), self.concatenate(red_word, all_vectors))
            if red_sim > best_sim:
                best_sim = red_sim
                similar_word = red_word

        return similar_word, best_sim
    
    # gets wn similarity using lin similarity (returns a dict: word -> similarity)
    # we can test different similarity metrics, but lin similarity is already implemented in the codebase
    def get_wn_similarity(self, word, red_words, bad_words):
        best_sim = 0.0
        similar_word = None

        # Get synsets for the given word (the potential clue)
        word_synsets = wordnet.synsets(word)

        for red_word in red_words:
            red_word_synsets = wordnet.synsets(red_word)
            for word_synset in word_synsets:
                for red_synset in red_word_synsets:
                    try:
                        score = word_synset.lin_similarity(red_synset, self.brown_ic)
                        if score and score > best_sim:
                            best_sim = score
                            similar_word = red_word
                    except:
                        continue
        
        return similar_word, best_sim

    
    

    # This method concatenates wordvectors from muliple sources (ie word2vec, gloVe) to create 1 vector
    def concatenate(self, word, wordvecs):
        concatenated = wordvecs[0][word]
        for vec in wordvecs[1:]:
            concatenated = np.hstack((concatenated, vec[word]))
        return concatenated



