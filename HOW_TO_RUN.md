# How To Run

Este guia assume que estas na raiz do repositorio.

## 1. Criar Ambiente Python

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

O projeto foi pensado para correr offline em Python. Nao e necessario ter ROS 2 instalado para correr os scripts principais de analise, desde que os dados da bag ja estejam disponiveis.

## 2. Converter Uma ROS 2 Bag

Exemplo:

```bash
python project_imp/scripts/rosbag_to_simple.py /path/to/rosbag2_file_or_folder \
  --output-dir project_imp/real_bag_main/data \
  --odom-topic /odom \
  --scan-topic /scan
```

Saidas:

- `odometry.csv` ou `raw_odometry.csv`
- `lidar_scans.npz`

## 3. Criar Mapa A Partir De Odometria + LiDAR

```bash
python project_imp/scripts/bag_to_map.py /path/to/rosbag2_file_or_folder \
  --output project_imp/results/real_bag_map \
  --odom-topic /odom \
  --scan-topic /scan \
  --resolution 0.05 \
  --max-range 5.0
```

Saidas:

- `.png` para visualizacao;
- `.pgm` para mapa de ocupacao;
- `.yaml` com metadata do mapa.

## 4. Extrair Landmarks Lineares

```bash
python project_imp/scripts/extract_line_landmarks.py \
  project_imp/maps/main_real_map.pgm \
  --map-yaml project_imp/maps/main_real_map.yaml \
  --output project_imp/landmarks/main_real_landmarks
```

Saidas:

- `main_real_landmarks.json`
- `main_real_landmarks.png`

## 5. Planear Rota No Mapa Simulado

```bash
python project_imp/scripts/plan_generic_route.py \
  --map-image project_imp/maps/floorplan_walkable_map.pgm \
  --map-yaml project_imp/maps/floorplan_walkable_map.yaml \
  --output project_imp/routes/floorplan_walkable_route
```

O script usa A* para encontrar uma rota entre targets definidos no mapa, evitando paredes e obstaculos.

## 6. Simular Odometria E LiDAR Com Ruido

Exemplo para gerar sensores simulados:

```bash
python project_imp/scripts/simulate_noisy_sensors.py \
  --map-image project_imp/maps/floorplan_walkable_map.pgm \
  --map-yaml project_imp/maps/floorplan_walkable_map.yaml \
  --route project_imp/routes/floorplan_walkable_route.csv \
  --output-dir project_imp/results/stable
```

Saidas principais:

- `true_path.csv` - trajetoria real simulada;
- `noisy_odometry.csv` - odometria com ruido acumulado;
- `lidar_scans.npz` - leituras LiDAR simuladas;
- `trajectory_overlay.png` - comparacao visual entre rota real e odometria.

## 7. Correr O EKF

Exemplo em dados simulados:

```bash
python project_imp/scripts/ekf_localization.py \
  --odom project_imp/results/stable/noisy_odometry.csv \
  --true project_imp/results/stable/true_path.csv \
  --lidar project_imp/results/stable/lidar_scans.npz \
  --map-image project_imp/maps/floorplan_walkable_map.pgm \
  --map-yaml project_imp/maps/floorplan_walkable_map.yaml \
  --landmarks project_imp/landmarks/floorplan_walkable_extracted_lines.json \
  --output-dir project_imp/results/stable_ekf
```

Saidas:

- `ekf_estimate.csv` - pose estimada pelo EKF;
- `ekf_overlay.png` - trajetoria real, odometria e EKF;
- `summary.json` - metricas e numero de updates aceites.

## 8. Reconstruir Mapa Com Poses Corrigidas Pelo EKF

```bash
python project_imp/scripts/sim_sensors_to_map.py \
  --odom project_imp/real_bag_main/results/ekf_estimate.csv \
  --lidar project_imp/real_bag_main/data/lidar_yaw_minus90.npz \
  --output project_imp/real_bag_main/results/mapping_ekf
```

Este passo cria um mapa usando as poses estimadas pelo EKF em vez da odometria crua.

## 9. Relocalizacao Global Com Particle Filter

Em simulacao:

```bash
python project_imp/scripts/hybrid_pf_to_ekf_test.py
```

Na bag real:

```bash
python project_imp/scripts/particle_filter_real_bag.py \
  --map-image project_imp/real_bag_main/map/main_real_map.pgm \
  --map-yaml project_imp/real_bag_main/map/main_real_map.yaml \
  --odom project_imp/real_bag_main/data/raw_odometry.csv \
  --lidar project_imp/real_bag_main/data/lidar_yaw_minus90.npz \
  --output-dir project_imp/real_bag_main/results/particle_filter_global
```

Nota: a relocalizacao global sem pose inicial aproximada e mais dificil em ambientes simetricos. O Particle Filter deve ser usado como mecanismo robusto para recuperar uma hipotese global, e o EKF continua a ser o localizador principal depois da pose ser aceite.

## 10. Imagens Para Relatorio/Apresentacao

As imagens mais importantes estao em:

```text
project_imp/images/
project_imp/real_bag_main/results/
project_imp/results/starter/
project_imp/results/stable/
project_imp/results/strong/
project_imp/results/very_strong/
```

