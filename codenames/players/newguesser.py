
import scipy.spatial.distance
import random
from operator import itemgetter
from nltk.corpus import wordnet

from players.guesser import Guesser

class AIGuesser(Guesser):

    def __init__(self, brown_ic=None, word_vectors=None):
        super().__init__()
        self.brown_ic = brown_ic
        self.word_vectors = word_vectors  # Use Word2Vec vectors
        self.num = 0
        self.weights = {'w2v': 0.5, 'wn': 0.5}  # Initial equal weights
        self.performance = {'w2v': [], 'wn': []}  # To track past performance

    def set_board(self, words):
        self.words = words

    def set_clue(self, clue, num):
        self.clue = clue
        self.num = num
        print("The clue is:", clue, num)

    def keep_guessing(self):
        return self.num > 0

    def get_answer(self):
        # Get results from both Word2Vec and WordNet methods
        w2v_results = self._compute_w2v_distance(self.clue, self.words)
        wn_results = self._compute_wordnet_similarity(self.clue, self.words)

        # Combine results using dynamic weights
        combined_results = self._combine_results(w2v_results, wn_results)
        
        # Adjust weights based on performance
        self._adjust_weights()

        # Select the best guess
        if not combined_results:
            return random.choice([word for word in self.words if word[0] != '*'])

        print(f'Guesses: {combined_results}')
        self.num -= 1
        return combined_results[0][1]

    def _compute_w2v_distance(self, clue, board):
        results = []
        for word in board:
            try:
                if word[0] == '*':
                    continue
                distance = scipy.spatial.distance.cosine(self.word_vectors[clue], self.word_vectors[word.lower()])
                results.append((distance, word))
            except KeyError:
                continue
        return results

    def _compute_wordnet_similarity(self, clue, board):
        results = []
        for word in board:
            for clue_synset in wordnet.synsets(clue):
                max_similarity = 0
                for word_synset in wordnet.synsets(word):
                    try:
                        similarity = clue_synset.lin_similarity(word_synset, self.brown_ic)
                        if similarity and similarity > max_similarity:
                            max_similarity = similarity
                    except:
                        continue
                if max_similarity > 0:
                    results.append((1 - max_similarity, word))  # Invert similarity for sorting
        return results

    def _combine_results(self, w2v_results, wn_results):
        # Normalize and weight the results based on current weights
        combined = {}
        for score, word in w2v_results:
            combined[word] = combined.get(word, 0) + self.weights['w2v'] * (1 - score)  # Normalize by inverting distance
        for score, word in wn_results:
            combined[word] = combined.get(word, 0) + self.weights['wn'] * (1 - score)  # Already normalized

        # Sort combined results by score
        sorted_combined = sorted(combined.items(), key=lambda x: -x[1])
        return [(score, word) for word, score in sorted_combined]

    def _adjust_weights(self):
        # Adjust weights based on performance - simplified: average past performance
        if self.performance['w2v'] and self.performance['wn']:
            w2v_avg = sum(self.performance['w2v']) / len(self.performance['w2v'])
            wn_avg = sum(self.performance['wn']) / len(self.performance['wn'])

            # Adjust weights to favor the method with higher average performance
            total = w2v_avg + wn_avg
            if total > 0:
                self.weights['w2v'] = w2v_avg / total
                self.weights['wn'] = wn_avg / total

    def update_performance(self, method, success):
        # Update performance tracking based on whether a guess was successful
        self.performance[method].append(1 if success else 0)