import random
from collections import deque, Counter

class Oracle:
  def __init__(self, k, seed):
    self.k = k
    self.seed = seed
    self.rng = random.Random(seed)
    self.history = deque(maxlen=k)
    self.table = {}

  def predict(self):
    context = tuple(self.history)
    random_list = ['f', 'd']
    if len(self.history) < self.k:
      return self.rng.choice(random_list)
    elif context not in self.table:
      return self.rng.choice(random_list)

    counts = self.table[context]
    if counts['f'] > counts['d']:
      return 'f'
    elif counts['d'] > counts['f']:
      return 'd'
    else:
      return self.rng.choice(random_list) 

  
  def update(self, key):
    context = tuple(self.history)
    if self.k == len(self.history):
      if context not in self.table:
        self.table[context] = Counter()
      self.table[context][key] += 1

    self.history.append(key)
