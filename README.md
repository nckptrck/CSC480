# Codenames AI Competition Framework

## Preamble

This project is an implementation for **CSC 480: Artificial Intelligence** at **California Polytechnic State University (Cal Poly)**. The project is developed by the following team members:

- **Robin Tun** 
- **Ashley Moreno**
- **Nick Patrick** 
- **Ellot Gerlach** 
- **Katie He** 

Instructor: **Dr. Rodrigo Canaan**

The project builds upon the [Codenames AI Competition Framework](https://github.com/CodenamesAICompetition/Game) on GitHub, which serves as the foundation for developing AI agents that can play the game "Codenames" by Vlaada Chvatil. This framework has been chosen because it emphasizes natural language understanding and communication, critical components for this AI competition.

## External Resources

This project leverages several external resources:

- **Python Libraries**: `gensim`, `nltk`, `colorama`
- **Competition Framework**: [Codenames AI Competition Framework](https://github.com/CodenamesAICompetition/Game)
- **Word Embedding Models**: Pre-trained models, including **WordNet** and **Word2Vec**, to enable semantic understanding for clue generation and guessing in the game.
- **Datasets**: Various corpora from NLTK, including the Brown Corpus.

## Submissions

Entrants in the competition will be able to submit up to two bots (at most 1 Codemaster and 1 Guesser).

## Prerequisite: Installation and Downloads

The installation of the [Anaconda Distribution](https://www.anaconda.com/distribution/) is recommended for managing dependencies easily. Installing NLTK and Gensim through conda is simpler and less time-consuming than other alternatives.

### Installation Steps

1. **Create and activate a new environment:**
    ```bash
    conda create --name codenames python=3.6
    conda activate codenames
    ```
2. **Install the required libraries:**
    ```bash
    conda install gensim
    pip install -U gensim nltk colorama
    ```
3. **Download NLTK data:**
    ```python
    python
    >>> import nltk
    >>> nltk.download('all')
    >>> exit()
    ```
4. **Clone the project repository:**
    ```bash
    git clone https://github.com/nckptrck/CSC480.git
    cd CSC480
    cd codenames
    ```
5. **Verify the installation:**
    ```bash
    python3 -c "import scipy, numpy, gensim.models.keyedvectors, argparse, importlib, nltk, nltk.corpus, nltk.stem"
    ```

### Alternative Installation Methods

You can use your system's package manager (e.g., `apt-get` on Debian, or `MacPorts/Homebrew` on macOS), or Python's `pip3`.

### Additional Resources

Download and set paths for the following:

- [Glove Vectors](https://nlp.stanford.edu/data/glove.6B.zip) (~2.25 GB)
- [Google News Vectors](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) (~3.5 GB)

## Running the Game from Terminal Instructions

To run the game, the terminal will require a specific order of arguments:

1. **Order of Arguments:**
    - `args[0]`: `run_game.py`
    - `args[1]`: `package.MyCodemasterClass`
    - `args[2]`: `package.MyGuesserClass`

For example, to run the AI bot using the combined model of **WordNet and Word2Vec** for the Codemaster and **Word2Vec** for the Guesser, use the following command:

```bash
python run_game.py players.codemaster_w2v_wn.AICodemaster players.guesser_w2v.AIGuesser --seed 3442 --w2v players/GoogleNews-vectors-negative300.bin --wordnet ic-brown.dat
```

**run_game.py simply handles system arguments then called game.Game().
See below for more details about calling game.Game() directly.**

Optionally if certain word vectors are needed, the directory to which should be specified in the arguments here.
5 argument parsers have been provided:
* --w2v *path/to/word_vectors*
  * (to be loaded by gensim)
* --glove *path/to/glove_vectors*
  *  (in stanford nlp format)
* --wordnet ic-brown.dat or ic-semcor.dat
  * (nltk corpus filename)

* --glove_cm *path/to/glove_vectors*
  * (legacy argument for glove_glove.py)
* --glove_guesser *path/to/glove_vectors*
  * (legacy argument for glove_glove.py)

An optional seed argument can be used for the purpose of consistency against the random library.
* --seed *Integer value* or "time"
  * ("time" uses Time.time() as the seed)

Other optional arguments include:
* --no_log
  * raise flag for suppressing logging
* --no_print
  * raise flag for suppressing printing to std out
* --game_name *String*
  * game_name in logfile

## Running the game from calling Game(...).run()

The class Game() that can be imported from game.Game is the main framework class.

An example of calling generalized vector codemaster and guesser from python code rather than command line
```
    cm_kwargs = {"vectors": [w2v, glove_50d, glove_100d], "distance_threshold": 0.3, "same_clue_patience": 1, "max_red_words_per_clue": 3}
    g_kwargs = {"vectors": [w2v, glove_50d, glove_100d]}
    Game(VectorCodemaster, VectorGuesser, seed=0, do_print=False,  game_name="vectorw2vglvglv03-vectorw2vglvglv", cm_kwargs=cm_kwargs, g_kwargs=g_kwargs).run()
```

See simple_example.py for an example of sharing word vectors,
passing kwargs to guesser/codemaster through Game,
and calling Game.run() directly.

## Game Class

The main framework class that calls your AI bots.

As mentioned above, a Game can be created/played directly by importing game.Game,
initializing with the args below, and calling the run() method.

```
Class that setups up game details and 
calls Guesser/Codemaster pair to play the game

Args:
    codemaster (:class:`Codemaster`):
        Codemaster (spymaster in Codenames' rules) class that provides a clue.
    guesser (:class:`Guesser`):
        Guesser (field operative in Codenames' rules) class that guesses based on clue.
    seed (int or str, optional): 
        Value used to init random, "time" for time.time(). 
        Defaults to "time".
    do_print (bool, optional): 
        Whether to keep on sys.stdout or turn off. 
        Defaults to True.
    do_log (bool, optional): 
        Whether to append to log file or not. 
        Defaults to True.
    game_name (str, optional): 
        game name used in log file. Defaults to "default".
    cm_kwargs (dict, optional): 
        kwargs passed to Codemaster.
    g_kwargs (dict, optional): 
        kwargs passed to Guesser.
```

## Codemaster Class
Any Codemaster bot is a python 3 class that derives from the supplied abstract base class Codemaster in `codemaster.py`.  The bot must implement three functions:
```
__init__(self)
set_game_state(words_on_board : List[str], key_grid : List[str]) -> None
get_clue() -> Tuple[str,int]
```
#### *details*

'__init__' **kwargs are passed through (can be used to pass pre-loaded word vectors to reduce load times for common NLP resources).  Some common examples are the Brown Corpus from NLTK's wordnet, the multi-dimensional GloVe vectors, and the 300 dimensional pre-trained Google NewsNewsBin word2vec vectors.

`set_game_state` is passed the list of words on the board, as well as the key grid provided to spymasters (codemasters).  The `words` are either: an all upper case word found in the English language or one of 4 special tokens: `'*Red*', '*Blue*', '*Civilian*', '*Assassin*'` indicating that the word that was originally at that location has been guessed and been found to be of that type.  The `key_grid` is a list of `'*Red*', '*Blue*', '*Civilian*', '*Assassin*'` indicating whether a spot on the board is on the team of the codemaster (`'*Red*'`), the opposing team (`'*Blue*'`), a civilian (`'*Civilian*'`), or the assassin (`'*Assassin*'`).


`get_clue` returns a tuple containing the clue, a single English word, and the number of words the Codemaster intends it to cover.

## Guesser Class

Any Guesser bot is a python 3 class that derives from the supplied abstract base class Guesser in `guesser.py`.  The bot must implement four functions:

```
__init__(self)
set_board(words: List[str]) -> None
set_clue(clue: str, num_guesses: int) -> None
keep_guessing -> bool
get_answer() -> Str
```

#### *details*

`__init__` is as above with the codemaster.

`set_board` is passed the list of words on the board.  The `words` are either: an all upper case word found in the English language or one of 4 special tokens: `'*Red*', '*Blue*', '*Civilian*', '*Assassin*'` indicating that the word that was originally at that location has been guessed and been found to be of that type.

`set_clue` is passed the clue and the number of guesses it covers, as supplied by the `get_clue` of the codemaster through the Game class.

`keep_guessing` is a function that the game engine checks to see if the bot chooses to keep guessing, as the bot must only make at least one guess, but may choose to guess until it has gone to the number supplied by get_clue + 1.

`get_answer` returns the current guess of the Guesser, given the state of the board and the previous clue.


## Rules of the Game

Codenames is a game focused on language understanding and communication. The competition is played with a single team format where both the **Codemaster** and **Guesser** are on the Red team. The goal is to identify all Red team's words as quickly as possible while avoiding incorrect guesses.

### Game Setup

- The board has 25 English words.
- The Codemaster knows which words belong to the Red team, Blue team, are civilians, or are the assassin.

### Codemaster's Role

- The Codemaster gives a clue (a single word) and a number representing the number of words related to that clue.  
  - For example: `('sky', 3)`.
- The clue must be semantically related to the Red team's words and must not derive from any words on the board.

### Guesser's Role

- The Guesser returns a list of guesses based on the clue, in order of confidence.
  - For example: `['CLOUD', 'FLIGHT', 'BIRD']`.
- Correct guesses help identify the Red team's words, but incorrect guesses (like guessing a civilian or Blue team's word) end the turn.

### Game Outcomes

The game ends when:

- All Red team's words are guessed — **team wins**.
- A Blue team's word or the assassin is guessed — **team loses**.

## Competition Rules

Competition results will be evaluated by the number of turns required to guess all 8 red words. Lower scores are better.

- **Instant Loss:** Guessing an assassin-linked word or all blue words will result in an immediate loss, with a score of 25 points.
- **Bot Pairing:** To ensure fair evaluation, Codemaster and Guesser bots will be paired with different bots in the competition.
