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
