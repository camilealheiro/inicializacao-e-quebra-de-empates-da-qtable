# Inicialização e Quebra de Empates da Q-table

A Aprendizagem por Reforço (RL) tem sido amplamente utilizada para resolver problemas complexos de tomada de decisão em ambientes dinâmicos. No entanto, os algoritmos de RL frequentemente são considerados "frágeis", ou seja, seu desempenho pode ser altamente sensível a pequenos ajustes nos hiperparâmetros ou na forma como são inicializados. Neste artigo, investigamos duas dessas influências: a inicialização da Q-table e a estratégia de desempate na escolha de ações "ε-greedy".   

## Objetivos
Este artigo busca responder a três questões principais:   
1. **Impacto da Inicialização da Q-table:** Como diferentes formas de inicialização (zero, pequenos valores positivos e valores pequenos aleatórios) afetam o desempenho dos algoritmos Monte Carlo e Q-Learning?
2. **Quebra de Empate no ε-Greedy:** Qual é o efeito de escolher ações com menor índice versus escolher aleatoriamente entre as melhores ações?
3. **Influência da Natureza do Ambiente:** O impacto dessas variações difere entre ambientes determinísticos e estocásticos?

## Metodologia
Para explorar essas questões, realizamos experimentos em dois ambientes do OpenAI Gym:
- **FrozenLake-v1:** Ambiente estocástico onde o agente pode escorregar ao se movimentar.
- **RaceTrack-v0:** Ambiente determinístico baseado em controle de velocidade de um carro.

Testamos os algoritmos Monte Carlo e Q-Learning, comparando três formas de inicialização da Q-table:
- **Zero:** Todas as entradas iniciam com 0 (sem conhecimento prévio).
- **Small Positive:** Pequenos valores positivos iniciais (incentiva exploração inicial).
- **Small Range:** Valores próximos de zero, positivos e negativos (neutralidade inicial).

Além disso, variamos a estratégia de desempate no ε-greedy:
- **Menor índice:** Sempre escolhe a primeira ação ótima (pode levar a viés).
- **Escolha aleatória:** Distribui chances igualmente entre ações ótimas (promove exploração).

Rodamos 5000 episódios para FrozenLake e 3000 episódios para RaceTrack, coletando estatísticas a cada 100 episódios. A escolha do número de episódios foi baseada no número mínimo para que o experimento gerasse uma estabilidade na soma de recompensas por episódio.   
Os resultados foram normalizados e agrupados em dois gráficos de linha para tendências ao longo do tempo para cada ambiente:
- **Comparação por inicialização:** Exibe o desempenho médio das três formas de inicialização, independentemente do algoritmo.
- **Comparação por quebra de desempate:** Compara o desempenho das estratégias de desempate, independentemente do algoritmo e inicialização.

Em seguida, foram gerados mais dois gráficos gerais de barras para comparação agregada que independem de algoritmo e ambiente:
- **Desempenho geral por inicialização:** Considera todas as variáveis e exibe a performance média de cada tipo de inicialização.
- **Desempenho geral por quebra de desempate:** Mostra o impacto geral das estratégias de desempate.

## Código
Para os algoritmos de Monte Carlo e Q-Learning, usamos os códigos de exemplos fornecidos pelo professor. A partir desses códigos algumas modificações simples foram feitas para o objetivo do projeto. Inicialmente, nas funções de escolha de ação de ambos os algoritmos, havia apenas a escolha do menor índice, mas como o objetivo era a comparação entre as duas formas de quebra de empate, acrescentamos a opção da escolha aleatória entre as melhores ações. Como acrescentamos o parâmetro random_tie_break para definir que ação seria escolhida, essa alteração também ocorreu nos algoritmos originais. Isso permitiu que tivéssemos gráficos comparativos entre as estratégias de desempate.

### Monte Carlo
```bash
# Monte Carlo
def choose_action(Q, state, num_actions, epsilon, random_tie_break=False):
    if np.random.random() < epsilon:
        return np.random.randint(0, num_actions)
    else:
        # Encontrar os índices das ações com maior valor Q
        best_actions = np.where(Q[state] == np.max(Q[state]))[0]

        if random_tie_break:
            return np.random.choice(best_actions)
        else:
            return best_actions[0]
```

### Q-learning
```bash
# Q-Learning
def epsilon_greedy(Q, state, epsilon, random_tie_break=False):
    num_actions = len(Q[state])
    if np.random.random() < epsilon:
        return np.random.randint(0, num_actions)
    elif random_tie_break:
        best_actions = np.where(Q[state] == np.max(Q[state]))[0]  # Ações ótimas
        return np.random.choice(best_actions)  # Escolha aleatória entre as melhores
    else:
        return np.argmax(Q[state])  # Escolha determinística
```
Além das alterações na escolha da ação, também adicionamos a opção de escolha dos tipos de inicialização no algoritmo, dessa forma init_type passou a ser chamado como parâmetro para que fosse possível gerar um gráfico comparativo entre esses índices:
```bash
if init_type == "zero":
        Q = np.zeros((env.observation_space.n, num_actions))
    elif init_type == "small_positive":
        Q = np.random.uniform(0, 0.01, (env.observation_space.n, num_actions))
    elif init_type == "small_range":
        Q = np.random.uniform(-0.01, 0.01, (env.observation_space.n, num_actions))
    else:
        raise ValueError("Invalid initialization type")
```

Desenvolvemos algumas funções para que o objetivo fosse atingido. Iniciamos com ```run_experiment``` que escolhe e executa o algoritmo (Monte Carlo ou Q-Learning), executa um experimento com um ambiente específico (FrozenLake ou RaceTrack), e testa as duas formas de desempate no ε-greedy. Além do algoritmo e ambiente, essa função também chama o tipo de inicialização entre "zero", "small positive", "small range", determina o número de episódios que serão rodados por ambiente e determina lr, gamma e epsilon, onde:
- lr é a taxa de aprendizado e através dela é controlado o quanto a Q-tabel é atualizada a cada iteração.
- gamma é o fator de desconto que define a importância das recompensas futuras.
- epsilon é a taxa de exploração que controla a chance de o agente explorar novas ações aleatoriamente em vez de seguir a melhor ação conhecida.

Em nosso experimento, optamos pelo lr moderado para que o agente aprenda sem alterar demais os valores da Q-table em cada atualização e o episilon baixo para priorizar as ações que já aprendeu. Já o gamma por ser um valor mais alto, incentiva o agente a pensar no longo prazo.
```bash
# Função para executar um experimento
def run_experiment(algorithm, env, init_type, episodes=1000, lr=0.1, gamma=0.95, epsilon=0.1):

    # Seleciona o algoritmo
    if algorithm == "montecarlo":
        greedy_results, _ = run_montecarlo2(env, episodes, lr, gamma, epsilon, init_type, random_tie_break=False, max_steps_per_episode=max_steps)
        random_results, _ = run_montecarlo2(env, episodes, lr, gamma, epsilon, init_type, random_tie_break=True, max_steps_per_episode=max_steps)

    elif algorithm == "qlearning":
        greedy_results, _ = run_qlearning(env, episodes, lr, gamma, epsilon, init_type, random_tie_break=False, max_steps_per_episode=max_steps)
        random_results, _ = run_qlearning(env, episodes, lr, gamma, epsilon, init_type, random_tie_break=True, max_steps_per_episode=max_steps)

    else:
        raise ValueError("Algoritmo não reconhecido.")
    env.close()

    return greedy_results, random_results
```

Depois disso criamos a função ```normalize_results``` para normalizar os valores de recompensas utilizando MinMaxScaler. Ela transforma os valores para ficarem entre 0 e 1, facilitando a comparação entre diferentes experimentos. Em seguida, o ```plot_comparison``` plota gráficos comparandultados de diferentes inicializações da Q-table (init_types). Para cada tipo de inicialização, plota duas curvas: Uma para a estratégia de desempate pelo menor índice e outra para a escolha aleatória entre as melhores ações. E por fim, ```load_results_from_csv``` lê um arquivo CSV contendo os resultados dos experimentos, converte as colunas de strings para listas de números (ast.literal_eval) e retorna um dicionário estruturado para facilitar a análise e a plotagem dos gráficos.
```bash
def normalize_results(results):
  scaler = MinMaxScaler()
  normalized_results = scaler.fit_transform(np.array(results).reshape(-1, 1)).flatten()
  return normalized_results

def plot_comparison(results, init_types, spacing=100):
  plt.figure(figsize=(12, 8))

  for init_type in init_types:
    greedy_results = results[init_type]["greedy"]
    random_results = results[init_type]["random"]

    greedy_normalized = normalize_results(greedy_results)
    random_normalized = normalize_results(random_results)

    indices = np.arange(0, len(greedy_normalized), spacing)
    greedy_spaced = greedy_normalized[indices]
    random_spaced = random_normalized[indices]

    plt.plot(indices, greedy_spaced, marker='o', linestyle='-', label=f"{init_type} (Menor Índice)")
    plt.plot(indices, random_spaced, marker='s', linestyle='--', label=f"{init_type} (Escolha Aleatória)")

  plt.xlabel("Episodes")
  plt.ylabel("Recompensas Normalizadas")
  plt.title("Comparação de Inicializações e Métodos de Desempate")
  plt.legend()
  plt.grid(True)
  plt.show()


def load_results_from_csv(filename):
    df = pd.read_csv(filename)
    results = {}

    for _, row in df.iterrows():
        init_type = row["init_type"]
        greedy_results = ast.literal_eval(row["greedy_results"])
        random_results = ast.literal_eval(row["random_results"])

        results[init_type] = {
            "greedy": greedy_results,
            "random": random_results
        }

    return results
```

Após a execução dessas funções, chamamos o código abaixo para cada ambiente em cada algoritmo. Neste caso, o código executa experimentos com Monte Carlo no ambiente “FrozenLake-v1”, testando três tipos de inicialização da Q-table. Ele coleta os resultados normalizados e os organiza em um Data Frame para análise. Em seguida, todos esses resultados são concatenados em um único csv.
```bash
results_lake = []
env = gym.make("FrozenLake-v1", is_slippery=True)

init_types = ["zero", "small_positive", "small_range"]
for init_type in init_types:
  greedy_results, random_results = run_experiment("montecarlo", env, init_type, episodes=5000, lr=0.1, gamma=0.95, epsilon=0.1)
  greedy_normalized = normalize_results(greedy_results)
  random_normalized = normalize_results(random_results)
  results_lake.append({
      "algorithm": "montecarlo",
      "env": "FrozenLake-v1",
      "init_type": init_type,
      "greedy_results": greedy_normalized,
      "random_results": random_normalized
  })

df_norm = pd.DataFrame(results_lake)
```

## Análise dos Resultados
Abaixo seguem os gráficos gerados pelos resultados dos experimentos. A imagem 1 representa a média dos desempenhos médios dos últimos 100 episódios de diferentes tipos de inicialização, considerando todos os ambientes e algoritmos. A imagem 2 segue a mesma proposta, porém avaliando a quebra de empate.   
IMAGEM   

