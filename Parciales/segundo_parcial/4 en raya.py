# %% [markdown]
# agente que sabe jugar 4 en raya

# %%
import numpy as np
import random
from tqdm import tqdm

class ConnectFour:
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.state = np.zeros((self.rows, self.cols))

    def valid_moves(self):
        return [c for c in range(self.cols) if self.state[0, c] == 0]

    def drop_piece(self, col, symbol):
        for row in range(self.rows-1, -1, -1):
            if self.state[row, col] == 0:
                self.state[row, col] = symbol
                return row, col
        raise ValueError("Movimiento ilegal")

    def is_game_over(self):
        # horizontal, vertical y diagonales
        for c in range(self.cols - 3):
            for r in range(self.rows):
                total = sum(self.state[r, c+i] for i in range(4))
                if abs(total) == 4: return np.sign(total)
        for c in range(self.cols):
            for r in range(self.rows - 3):
                total = sum(self.state[r+i, c] for i in range(4))
                if abs(total) == 4: return np.sign(total)
        for c in range(self.cols - 3):
            for r in range(self.rows - 3):
                total = sum(self.state[r+i, c+i] for i in range(4))
                if abs(total) == 4: return np.sign(total)
        for c in range(self.cols - 3):
            for r in range(3, self.rows):
                total = sum(self.state[r-i, c+i] for i in range(4))
                if abs(total) == 4: return np.sign(total)
        if np.all(self.state != 0):
            return 0  # Empate
        return None  # Sigue jugando

    def reset(self):
        self.state = np.zeros((self.rows, self.cols))


# %%
class Agent:
    def __init__(self, symbol, alpha=0.1, epsilon=0.1):
        self.symbol = symbol
        self.alpha = alpha
        self.epsilon = epsilon
        self.value_function = {}
        self.history = []

    def reset(self):
        self.history = []

    def get_state_hash(self, board):
        return str(board.state.reshape(-1))

    def choose_action(self, board):
        moves = board.valid_moves()
        if random.random() < self.epsilon:
            return random.choice(moves)  # Exploración
        best_value = -float('inf')
        best_move = moves[0]
        for move in moves:
            temp_board = board.state.copy()
            for r in range(board.rows-1, -1, -1):
                if temp_board[r, move] == 0:
                    temp_board[r, move] = self.symbol
                    break
            state_hash = str(temp_board.reshape(-1))
            value = self.value_function.get(state_hash, 0.5)
            if value > best_value:
                best_value = value
                best_move = move
        return best_move

    def update_history(self, board):
        self.history.append(self.get_state_hash(board))

    def update_value_function(self, reward):
        target = reward
        for state in reversed(self.history):
            value = self.value_function.get(state, 0.5)
            self.value_function[state] = value + self.alpha * (target - value)
            target = self.value_function[state]


# %%
class Game:
    def __init__(self, p1, p2):
        self.board = ConnectFour()
        self.p1 = p1
        self.p2 = p2
        self.p1.symbol = 1
        self.p2.symbol = -1

    def play(self, episodes=10000):
        wins = {1: 0, -1: 0, 0: 0}
        for _ in tqdm(range(episodes)):
            self.board.reset()
            self.p1.reset()
            self.p2.reset()
            players = [self.p1, self.p2]
            turn = 0
            while True:
                player = players[turn % 2]
                move = player.choose_action(self.board)
                self.board.drop_piece(move, player.symbol)
                self.p1.update_history(self.board)
                self.p2.update_history(self.board)
                result = self.board.is_game_over()
                if result is not None:
                    self.p1.update_value_function(1 if result == self.p1.symbol else 0.5 if result == 0 else 0)
                    self.p2.update_value_function(1 if result == self.p2.symbol else 0.5 if result == 0 else 0)
                    wins[result] += 1
                    break
                turn += 1
        return wins


# %%
import pickle
# Entrenamiento
agent1 = Agent(symbol=1)
agent2 = Agent(symbol=-1)

game = Game(agent1, agent2)
resultados = game.play(episodes=3000)
print(resultados)

print("Resultados tras entrenamiento:")
print(f"Victorias agente 1: {resultados[1]}")
print(f"Victorias agente 2: {resultados[-1]}")
print(f"Empates: {resultados[0]}")

# Guardar Q-table
with open('qtable_connect4.pkl', 'wb') as f:
    pickle.dump(agent1.value_function, f)

# %%
import pandas as pd

# Ordenar la función de valor de mayor a menor
funcion_de_valor = sorted(agent1.value_function.items(), key=lambda kv: kv[1], reverse=True)

# Crear un DataFrame con los estados y sus valores
tabla = pd.DataFrame({
    'estado': [x[0] for x in funcion_de_valor],
    'valor': [x[1] for x in funcion_de_valor]
})

print(tabla)


# %%
from matplotlib import pyplot as plt

ax = tabla['valor'].plot(kind='hist', bins=20, title='valor')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()


# %%
from matplotlib import pyplot as plt

ax = tabla['valor'].plot(kind='line', figsize=(8, 4), title='valor')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()



