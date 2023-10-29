import gym
import numpy as np
from copy import deepcopy
from gym import spaces
import pygame
import pandas as pd

class Wordle:
    def __init__(self, word, rows=6, letters=5):
        self.g_count = 0
        self.word = word
        self.w_hash_table = {}
        if word is not None:
            for x, l in enumerate(word):
                if l in self.w_hash_table:
                    self.w_hash_table[l]['count'] += 1
                    self.w_hash_table[l]['pos'].append(x)
                else:
                    self.w_hash_table[l] = {'count':1, 'pos':[x]}
        self.rows = rows
        self.letters = letters
        self.board = [['' for _ in range(letters)] for _ in range(rows)]
        self.colours = [['B' for _ in range(letters)] for _ in range(rows)]
        self.alph = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

    def is_end(self):
        if self.board[-1] != ['' for _ in range(self.letters)]:
            return True
        else:
            r = self.game_result()
            if r[0] == True:
                return True
            else:
                return False

    def game_result(self):
        win = (False, 100)
        for i, r in enumerate(self.board):
            if self.word == ''.join(r):
                win = (True, i)
                break
        return win

    def update_board(self, u_inp):
        w_hash_table = deepcopy(self.w_hash_table)
        i_hash_table = {}
        for x, l in enumerate(str(u_inp).upper()):
            self.board[self.g_count][x] = l
            if l in i_hash_table:
                i_hash_table[l].append(x)
            else:
                i_hash_table[l] = [x]
        colours = {'G':[],'B':[],'Y':[]}
        for l in i_hash_table:
            if l in w_hash_table:
                g_hold = []
                for p in i_hash_table[l]:
                    if p in w_hash_table[l]['pos']:
                        g_hold.append(p)
                for p in g_hold:
                    i_hash_table[l].remove(p)
                colours['G'] += g_hold
                if len(g_hold) < w_hash_table[l]['count']:
                    y_hold = []
                    for p in i_hash_table[l]:
                        y_hold.append(p)
                        if len(y_hold) == w_hash_table[l]['count']:
                            break
                    for p in y_hold:
                        i_hash_table[l].remove(p)
                    colours['Y'] += y_hold
                for p in i_hash_table[l]:
                    colours['B'].append(p)
            else:
                colours['B'] += i_hash_table[l]
                i_hash_table[l] = []
        for c in colours:
            for p in colours[c]:
                self.colours[self.g_count][p] = c
        self.g_count += 1

    def valid_guess(self, u_inp):
        if len(u_inp) == 5 and False not in [False for s in str(u_inp).upper() if s not in self.alph]:
            return True
        else:
            return False

class WordleEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    SCREEN_DIM = 500
    GREEN = "#6aaa64"
    YELLOW = "#c9b458"
    GREY = "#787c7e"
    OUTLINE = "#d3d6da"
    FILLED_OUTLINE = "#878a8c"

    def __init__(self, answers, logging=False):
        self.logging = logging
        self.answers = pd.DataFrame(answers)
        self.answers.columns = ['words']
        self.screen = None
        self.isopen = False
        self.GUESSES = 6
        self.LETTERS = 5
        self.WORD = self.answers['words'].sample(n=1).tolist()[0].upper()
        self.WORDLE = Wordle(self.WORD, self.GUESSES, self.LETTERS)
        self.alpha = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
        self.colors = ['B', 'Y', 'G']
        self.is_game_over = False
        self.guessed_words = []
        self.blank_letters = []

        # our action space is the total amount of possible words to guess
        self.action_space = spaces.Discrete(len(answers))
        #our observation space is the current wordle board in form of (letter, color) with 5x6 (5 letters, 6 guesses)
        #modified to work with gym/baselines
        #same thing basically, only 0-26 is '' to z and 27-29 is B, Y, G
        # first 6 rows are guesses and last 6 rows are colors
        # changed shape to be 3 dimensions so that we can apply conv2d layers to it
        # at some point we should try to normalize the obs space
        # since right now its on a 0-29 scale instead of a 0-1.
        self.observation_space = spaces.Box(low=0, high=29, shape=(1,12,5), dtype='int32')
        self.current_episode = -1
        self.episode_memory: list[any] = []

    def step(self, action):
        if self.is_game_over:
            return RuntimeError('Episode is already done')
        guess = self._take_action(action)
        reward = self._get_reward(guess)
        self.guessed_words.append(guess.upper())
        observation = self._get_observation()
        res = self.WORDLE.colours[self.WORDLE.g_count-1]
        self.blank_letters.extend([ l for i,l in enumerate(guess) if res[i] == 'B' and l not in self.blank_letters])
        return observation, reward, self.is_game_over, {}

    def reset(self):
        self.current_episode = -1
        self.episode_memory.append([])
        self.is_game_over = False
        self.WORD = self.answers['words'].sample(n=1).tolist()[0].upper()
        self.WORDLE = Wordle(self.WORD, self.GUESSES, self.LETTERS)
        self.guessed_words = []
        self.blank_letters = []
        if self.logging:
            #print(self.WORDLE.word)
            pass
        self.close()
        return self._get_observation()

    def render(self, mode='human'):
        pygame.init()
        pygame.display.init()
        self.screen = pygame.display.set_mode((self.SCREEN_DIM, self.SCREEN_DIM))
        font = pygame.font.Font('freesansbold.ttf', 30)
        for col in range(0, 5):
            for row in range(0, 6):
                pygame.draw.rect(self.screen, self.OUTLINE, [col * 100 + 12, row * 100 + 12, 75, 75], 3, 5)
                color = self.GREEN if self.WORDLE.colours[row][col] == 'G' else self.YELLOW if self.WORDLE.colours[row][col] == 'Y' else self.GREY
                piece_text = font.render(self.WORDLE.board[row][col], True, color)
                self.screen.blit(piece_text, (col * 100 + 30, row * 100 + 25))
        #pygame.draw.rect(screen, self.GREEN, [5, turn * 100 + 5, WIDTH - 10, 90], 3, 5)
        if mode == "human":
            pygame.event.pump()
            pygame.display.flip()             
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    def _take_action(self, action):
        # turn action into guess
        guess = self.answers['words'][action].upper()
        self.episode_memory[self.current_episode].append(guess)
        if self.logging:
            print(guess)
            pass
        self.WORDLE.update_board(guess)
        res = self.WORDLE.colours[self.WORDLE.g_count-1]
        self.is_game_over = self.WORDLE.word == guess.upper() or self.WORDLE.g_count == self.GUESSES
        
        if self.is_game_over and self.logging:
            print(f'Guessed in : {len(self.guessed_words)} \nWords: ', end='')
            print(*self.guessed_words, sep=",")
            print(f'Answer: {self.WORD}')
        if self.WORDLE.word == guess:
            print(f'Guessed {self.WORDLE.word} in {self.WORDLE.g_count} guesses!')
        return guess
    def _get_reward(self, guess):
        result, tries = self.WORDLE.game_result()
        rewards = np.zeros(5)
        #heavily penealize guessing the same word multiple times
        #If a word isn't the right guess, we shouldn't guess it again
        #could do the same thing for letters, as if a letter is blank(grey)
        # then the only reason to use a word with a letter in it
        # is to check other letter posistions
        #so it shouldn't be a heavy penalty but it should be a penalty
        checked_guess = []
        for g, c in zip(guess, self.WORD):
            if g == c:
                checked_guess.append('G')
            elif g in c:
                checked_guess.append('Y')
            else:
                checked_guess.append('B')
        for i,c in enumerate(checked_guess):
            if c == self.colors[2]:
                rewards[i] = 3
            elif c == self.colors[1]:
                rewards[i] = 2
            elif c == self.colors[0]:
                rewards[i] = 1
        #check guesses up to and including our current guess

        reward = np.sum(rewards)
        '''
        for g in range(self.WORDLE.g_count):
            word = self.WORDLE.board[g]
            current = ''.join(word)
            if guess in self.guessed_words:
                print('did this')
                return 0
                print('0')
            for l in word: 
                if l in self.blank_letters:
                    reward -= 0.3'''
        if guess in self.guessed_words:
            print('0')
            return 0
        for l in guess:
            if l in self.blank_letters:
                reward -= 1.3
        if self.logging:
            print(self.WORD)
            print(rewards)
            print(reward)
        return reward

    def _get_observation(self):
        board = np.array(self.WORDLE.board) #2d array of 5x6
        colors = np.array(self.WORDLE.colours) #2d array of 5x6
        results = np.vstack((board, colors)) #stacks boards and colors by rows resulting in a 2d array of 5x12
        convertletterstonum = lambda letter: [self.alpha.index(l) + 1 if l in self.alpha else 0 for l in letter]
        convertcolortonum = lambda color: [self.colors.index(c)+27 for c in color]
        guesses = np.array([convertletterstonum(l) if i <=5 else convertcolortonum(l) for i, l in enumerate(results)])
        guesses3d = np.expand_dims(guesses, axis=0)
        if self.logging:
            pass
            #print(np.shape(guesses))
            #print(np.shape(guesses3d))
        return guesses3d