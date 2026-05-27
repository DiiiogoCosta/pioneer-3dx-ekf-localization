# Explicação rápida do código

## Scripts principais

### `rosbag_to_simple.py`

Extrai das bags os dados mínimos que precisamos:

- odometria para CSV;
- scans LiDAR para NPZ.

Isto evita precisar de ROS no Mac para o resto do processamento.

### `bag_to_map.py` / `bag_to_hit_map.py`

Projetam os pontos do LiDAR no mundo usando a odometria. O objetivo é criar um mapa inicial, mesmo que fique ruidoso.

### `clean_occupancy_map.py`

Limpa o mapa removendo componentes pequenas e juntando paredes próximas. Serve para reduzir ruído de LiDAR.

### `extract_line_landmarks.py`

Lê o mapa e encontra segmentos retos nas fronteiras ocupadas. Usa uma lógica tipo RANSAC:

1. escolhe pares de pontos;
2. tenta ajustar uma reta;
3. conta pontos próximos dessa reta;
4. guarda linhas longas e consistentes.

No mapa final extraiu 16 landmarks lineares.

### `plan_generic_route.py`

Planeia uma rota sem colisão com A* numa grelha de ocupação.

Antes de planear, infla os obstáculos com o raio do robô + margem de segurança. Assim o caminho não passa demasiado perto das paredes.

### `simulate_noisy_sensors.py`

Gera uma simulação do robô:

- trajetória real;
- odometria com erro acumulado;
- LiDAR com ruído e dropout.

Os perfis `starter`, `stable`, `strong` e `very_strong` mudam estes parâmetros de ruído.

### `ekf_localization.py`

É o script principal do EKF.

Estado estimado:

```text
x, y, theta
```

O algoritmo faz:

1. previsão com odometria;
2. deteção de linhas no scan LiDAR;
3. associação das linhas detetadas com landmarks do mapa;
4. update EKF com Kalman Gain;
5. guarda a trajetória corrigida.

### `sim_sensors_to_map.py`

Reconstrói um mapa usando scans LiDAR e uma trajetória.

Usámos este script duas vezes:

- com odometria ruidosa, para mostrar mapa deformado;
- com poses corrigidas pelo EKF, para mostrar mapa melhor.

## Ideia que convém saber explicar

A odometria é boa localmente, mas acumula erro. O LiDAR observa paredes/linhas do ambiente. O EKF combina as duas fontes:

- odometria dá movimento previsto;
- LiDAR corrige a pose quando encontra landmarks conhecidos.

Assim, mesmo que a odometria derive, o EKF consegue manter a localização alinhada com o mapa.
