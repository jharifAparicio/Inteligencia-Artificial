# %%
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
from collections import defaultdict

# %% [markdown]
# # Método 1: Epsilon-Greedy

# %% [markdown]
# Método 1: Epsilon-Greedy aplicado al entorno Blackjack-v1 de Gymnasium
# 
# Descripción:
# Se utiliza el algoritmo epsilon-greedy para aprender una política óptima que maximice la recompensa
# esperada en el juego de Blackjack. En cada paso, el agente decide si explora (elige una acción aleatoria)
# o explota (elige la mejor acción conocida hasta ahora). Se usa una tabla Q para almacenar los valores
# 
# estimados de cada acción en cada estado.
# 
# Entorno:
# - Nombre: Blackjack-v1
# - Acciones: 0 = Stick (quedarse), 1 = Hit (pedir carta)
# - Observaciones: (suma del jugador, carta visible del dealer, as usable)

# %%
# Creamos el entorno de Blackjack
env = gym.make("Blackjack-v1")

# %% [markdown]
# Sí, usar sab=True en el entorno de Blackjack suele resultar en un aprendizaje más realista y estable para el método epsilon-greedy, porque:
# 
# Las recompensas son más informativas (Blackjack natural da recompensa 1.5, no solo 1).
# 
# Las reglas del dealer (plantarse en soft 17) hacen que la dinámica del juego sea más consistente con la realidad.
# 
# Esto ayuda a que el agente aprenda una política más óptima y cercana a estrategias reales.
# 
# En resumen:
# Con sab=True, el método epsilon-greedy suele funcionar mejor porque el entorno es más representativo del Blackjack verdadero y las recompensas reflejan mejor las ganancias.

# %%
# Función para crear la Q-table como diccionario de valores por defecto
def create_Q():
    # Cada estado tendrá un array de 2 valores (para acciones: 0 = quedarte, 1 = pedir carta)
    return defaultdict(lambda: np.zeros(env.action_space.n))
# q = {1,2,3,4,5}

# %%
# Política epsilon-greedy para seleccionar acciones
def epsilon_greedy(Q, state, epsilon):
    """
    Selecciona una acción según la política epsilon-greedy.
    - Con probabilidad epsilon elige una acción aleatoria (exploración).
    - Con probabilidad 1-epsilon elige la mejor acción conocida (explotación).
    """
    if np.random.rand() < epsilon:
        return env.action_space.sample()  # Acción aleatoria
    else:
        return np.argmax(Q[state])  # Mejor acción según Q

# %%
def train_q_learning_multi_epsilon(env, epsilons, n_episodes=100_000, alpha=0.1, gamma=1.0):
    """
    Entrena agentes con Q-learning en el entorno Blackjack con diferentes epsilons.

    Parámetros:
    - env: entorno de Blackjack (gym).
    - epsilons: lista de valores epsilon a probar.
    - n_episodes: número de partidas para entrenar cada agente.
    - alpha: tasa de aprendizaje.
    - gamma: factor de descuento.

    Retorna:
    - Q_tables: lista de Q-tables entrenadas, una por cada epsilon.
    - recompensas_medias: matriz (len(epsilons) x n_episodes) con recompensas por episodio.
    """
    Q_tables = []
    recompensas_medias = np.zeros((len(epsilons), n_episodes))

    for idx, epsilon in enumerate(epsilons):
        Q = create_Q()  # función para crear Q-table vacía
        for ep in range(n_episodes):
            state, _ = env.reset()
            done = False
            recompensa_ep = 0

            while not done:
                action = epsilon_greedy(Q, state, epsilon)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                best_next_action = np.argmax(Q[next_state])
                td_target = reward + gamma * Q[next_state][best_next_action]
                Q[state][action] += alpha * (td_target - Q[state][action])

                state = next_state
                recompensa_ep += reward

            recompensas_medias[idx, ep] = recompensa_ep

        Q_tables.append(Q)

    return Q_tables, recompensas_medias


# %% [markdown]
# # Uso
# Entrenamos y mostramos la politica aplicada

# %%
epsilons = [0, 0.1, 0.01, 0.3]
n_episodes = 100_000

Q_tables, recompensas_medias = train_q_learning_multi_epsilon(env, epsilons, n_episodes=n_episodes)

# %%
plt.figure(figsize=(30,10))
for i, e in enumerate(epsilons):  # Excluimos epsilon=0 para no mostrarlo en el gráfico
    # Promediamos recompensas cada 1000 episodios para suavizar gráfico
    medias_suavizadas = np.convolve(recompensas_medias[i], np.ones(1000)/1000, mode='valid')
    plt.plot(medias_suavizadas, label=f'$\epsilon$ = {e}')
plt.legend()
plt.grid(True)
plt.xlabel('Episodios (x1000)')
plt.ylabel('Recompensa media')
plt.title('Recompensa media por episodio y epsilon')
plt.show()

# %% [markdown]
# # Metodo 2: Implementacion incremental

# %% [markdown]
# Este método simula muchas partidas donde un agente aprende a elegir la mejor acción (pedir carta o quedarse) en diferentes estados del juego usando un enfoque incremental para actualizar los valores de acción (Q-values):
# 
# Se usa un valor Q[a] para cada acción 
# 𝑎
# a que representa la recompensa esperada al tomar esa acción.
# 
# En cada paso, el agente elige una acción:
# 
# Con probabilidad 
# 𝜖
# ϵ explora: elige una acción aleatoria.
# 
# Con probabilidad 
# 1
# −
# 𝜖
# 1−ϵ explota: elige la acción con mayor Q-value.
# 
# Luego recibe una recompensa (ganancia o pérdida del juego) y actualiza el Q-value de esa acción usando la fórmula incremental:
# 
# 𝑄
# [
# 𝑎
# ]
# ←
# 𝑄
# [
# 𝑎
# ]
# +
# 𝛼
# ×
# (
# 𝑟
# 𝑒
# 𝑐
# 𝑜
# 𝑚
# 𝑝
# 𝑒
# 𝑛
# 𝑠
# 𝑎
# −
# 𝑄
# [
# 𝑎
# ]
# )
# Q[a]←Q[a]+α×(recompensa−Q[a])
# Esto va ajustando poco a poco la expectativa del valor de cada acción.
# 
# Finalmente, se promedian las recompensas y las veces que se tomó la mejor acción para ver el desempeño.

# %%
env_Increment = gym.make("Blackjack-v1", sab=True)

# %%
# hiperparámetros
partidas = 100_000
alpha = 0.5 # tasa de aprendizaje
epsilons = [0., 0.01, 0.1, 0.3]
turnos = 100  # número de experiencias para evaluar progreso

# %%
# Mejor opción para Q-table con defaultdict
def create_Q():
    # Cada estado tiene un array de 2 elementos: [Q(s, quedarse), Q(s, pedir carta)]
    return defaultdict(lambda: np.zeros(env.action_space.n))

# %%
recompensas_medias = np.zeros((len(epsilons), turnos))
acciones_optimas = np.zeros((len(epsilons), turnos))

for ej in range(partidas):
    for i, epsilon in enumerate(epsilons):
        Q = create_Q()  # Q-table inicializada con defaultdict
        state, _ = env_Increment.reset()
        done = False
        paso = 0

        while not done and paso < turnos:
            key_state = state  # en tu entorno Blackjack el estado ya es hashable

            # epsilon-greedy
            if np.random.rand() < epsilon:
                action = env_Increment.action_space.sample()
            else:
                q_values = Q[key_state]
                action = np.argmax(q_values)

            next_state, reward, terminated, truncated, _ = env_Increment.step(action)
            done = terminated or truncated

            # actualización incremental Q-learning
            old_q = Q[key_state][action]
            Q[key_state][action] += alpha * (reward - old_q)

            recompensas_medias[i][paso] += reward
            acciones_optimas[i][paso] += (action == np.argmax(Q[key_state]))

            state = next_state
            paso += 1

recompensas_medias /= partidas
acciones_optimas /= partidas

# %%
plt.figure(figsize=(10,8))
for i, e in enumerate(epsilons):  # Excluimos epsilon=0 para no mostrar la línea plana
    plt.plot(recompensas_medias[i], label=f'$\epsilon$ = {e}')
plt.legend()
plt.grid(True)
plt.xlabel('Experiencias')
plt.ylabel('Recompensa media')
plt.title('Recompensa media por experiencia y epsilon')
plt.show()


# %% [markdown]
# # Metodo 3: Valores iniciales optimistas

# %% [markdown]
# Este método simula el entrenamiento de un agente que aprende a jugar Blackjack mediante un algoritmo similar a Q-learning, pero simplificado:
# 
# Estados: en Blackjack, el estado es la situación actual del jugador (p. ej., suma de cartas, si tiene un as usable, carta visible del dealer). Aquí se simula con un conjunto discreto simplificado para el ejemplo.
# 
# Acciones: en Blackjack solo hay dos acciones posibles:
# 
# 0: quedarse (stand)
# 
# 1: pedir carta (hit)
# 
# Q-values: son valores que estiman la recompensa esperada al tomar una acción en un estado dado.
# 
# Epsilon-greedy: estrategia para balancear exploración (elegir acción aleatoria con probabilidad epsilon) y explotación (elegir acción con mejor Q-valor).
# 
# Alpha: tasa de aprendizaje para actualizar el Q-valor de una acción.
# 
# Recompensas: pueden ser -1 (pierde), 0 (empate), 1 (gana), simuladas con la función q o extraídas directamente del entorno de Blackjack.
# 
# La simulación corre partidas episodios, y en cada uno se simulan hasta turnos acciones o hasta que termine la partida.

# %%
env_Valors_initials = gym.make('Blackjack-v1', sab=True)

# %%
partidas = 100_000
turnos = 100
alpha = 0.5
epsilons = [0, 0.1, 0.01, 0.3]

recompensas_medias = np.zeros((len(epsilons), turnos))
acciones_optimas = np.zeros((len(epsilons), turnos))

for ej in range(partidas):
    for i, epsilon in enumerate(epsilons):
        # Q con claves (estado, acción), inicializamos a 0 o 5 si epsilon=0 (optimista)
        if epsilon == 0:
            Q = defaultdict(lambda: 5.0)
        else:
            Q = defaultdict(lambda: 0.0)

        state, _ = env_Valors_initials.reset()
        done = False
        paso = 0

        while not done and paso < turnos:
            # Epsilon-greedy
            if np.random.uniform() < epsilon:
                action = env_Valors_initials.action_space.sample()
            else:
                q0 = Q[(state, 0)]
                q1 = Q[(state, 1)]
                action = 0 if q0 >= q1 else 1

            next_state, reward, terminated, truncated, _ = env_Valors_initials.step(action)
            done = terminated or truncated

            # Actualización Q simple (sin gamma)
            old_q = Q[(state, action)]
            Q[(state, action)] = old_q + alpha * (reward - old_q)

            recompensas_medias[i][paso] += reward

            # Acción óptima (según Q en estado actual)
            q0 = Q[(state, 0)]
            q1 = Q[(state, 1)]
            mejor_accion = 0 if q0 >= q1 else 1
            acciones_optimas[i][paso] += (action == mejor_accion)

            state = next_state
            paso += 1

recompensas_medias /= partidas
acciones_optimas /= partidas

# %%
plt.figure(figsize=(8,5))
for i, epsilon in enumerate(epsilons):
    plt.plot(recompensas_medias[i], label=f'$\epsilon$ = {epsilon}')
plt.xlabel('Turnos')
plt.ylabel('Recompensa media')
plt.title('Recompensa media con valores iniciales optimistas en Blackjack')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
for i, epsilon in enumerate(epsilons):
    plt.plot(acciones_optimas[i], label=f'$\epsilon$ = {epsilon}')
plt.xlabel('Turnos')
plt.ylabel('Proporción acciones óptimas')
plt.title('Proporción de acciones óptimas por epsilon en Blackjack')
plt.legend()
plt.grid(True)
plt.show()


# %% [markdown]
# # Metodo 4: 

# %%
def softmax(x):
    e_x = np.exp(x - np.max(x))  # para estabilidad numérica
    return e_x / np.sum(e_x)

# %%
env_softmax = gym.make('Blackjack-v1', sab=True)
partidas = 1000
turnos = 100
alphas = [0.1, 0.01 ,0.4]
recompensas_medias = np.zeros((len(alphas), turnos))
acciones_optimas = np.zeros((len(alphas), turnos))

# %%
for ej in range(partidas):
    for i, alpha in enumerate(alphas):
        H = np.zeros(2)  # Preferencias para acciones [0, 1]
        pi = softmax(H)
        recompensas = []
        state, _ = env_softmax.reset()
        done = False
        paso = 0

        while not done and paso < turnos:
            a = np.random.choice([0,1], p=pi)  # acción según softmax
            next_state, reward, terminated, truncated, _ = env_softmax.step(a)
            done = terminated or truncated
            recompensas.append(reward)
            media_recompensa = np.mean(recompensas)

            # Actualizar preferencias (softmax gradient)
            for j in range(2):
                if j == a:
                    H[j] += alpha * (reward - media_recompensa) * (1 - pi[j])
                else:
                    H[j] -= alpha * (reward - media_recompensa) * pi[j]

            pi = softmax(H)

            # Acción óptima: la que maximizó el reward actual
            mejor_accion = 0 if reward == max(reward, -1, 0, 1) else 1
            recompensas_medias[i][paso] += reward
            acciones_optimas[i][paso] += (a == mejor_accion)

            state = next_state
            paso += 1


# %%
recompensas_medias /= partidas
acciones_optimas /= partidas

# %%
# Gráfico de recompensa media
plt.figure(figsize=(15,10))
for i, a in enumerate(alphas):
    plt.plot(recompensas_medias[i], label=fr'$\alpha$ = {a}')
plt.legend()
plt.grid(True)
plt.xlabel('Turnos')
plt.ylabel('Recompensa media')
plt.title('Softmax Gradient Bandit en Blackjack')
plt.show()

# Gráfico de proporción de acciones óptimas
plt.figure(figsize=(15,10))
for i, a in enumerate(alphas):
    plt.plot(acciones_optimas[i], label=fr'$\alpha$ = {a}')
plt.legend()
plt.grid(True)
plt.xlabel('Turnos')
plt.ylabel('Proporción acción óptima')
plt.title('Proporción de acciones óptimas (Softmax) en Blackjack')
plt.show()



