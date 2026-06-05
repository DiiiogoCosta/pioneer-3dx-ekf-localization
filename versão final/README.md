# Versao Final - EKF Localization Pioneer 3-DX

Esta pasta contem apenas a versao final defensavel do projeto. As tentativas antigas com mapas manuais, landmarks de paredes, grids grandes e testes intermédios ficaram fora desta pasta.

## Pipeline final

```text
Bag -> slam_toolbox -> mapa 2D
Bag -> odometria + LiDAR
Mapa + LiDAR -> correlative scan matching
Odometria -> prediction do EKF
Scan matching -> correction do EKF
Pose da slam_toolbox -> comparacao, nao entrada do EKF
```

## Conteudo

- `scripts/rosbag_to_simple.py`: converte uma rosbag ROS2 para ficheiros simples (`odometry.csv` e LiDAR `.npz`) para correr em Python fora do ROS.
- `scripts/extract_slam_tf_pose.py`: extrai a pose estimada pela `slam_toolbox` a partir de `/tf`, usada apenas para comparacao.
- `scripts/ekf_correlative_scanmatch.py`: algoritmo final. Usa odometria na predicao do EKF e usa scan matching contra o mapa para gerar uma medicao absoluta de pose.
- `maps/corredor`: mapa criado pela `slam_toolbox` para o corredor.
- `maps/sala2`: mapa criado pela `slam_toolbox` para a sala.
- `data/corredor`: odometria, LiDAR e pose SLAM do corredor.
- `data/sala2`: odometria, LiDAR e pose SLAM da sala. A pose SLAM da sala e considerada incerta.
- `results/corredor`: resultado principal, com boa ground truth.
- `results/sala2`: resultado secundario e diagnostico de ground truth possivelmente desalinhada.

## Resultado principal

O corredor e o caso usado como resultado principal:

- odometria: erro medio `0.986 m`, erro final `1.325 m`
- EKF + scan matching: erro medio `0.179 m`, erro final `0.097 m`

Na imagem final:

- verde: pose da `slam_toolbox`, usada como referencia de comparacao
- vermelho: odometria
- azul: EKF corrigido

Imagem: `results/corredor/overlay_ekf_vs_odom_vs_slam.png`

## Sala

A sala foi mantida como caso de limitacao. A comparacao numerica nao e tao confiavel porque a pose da `slam_toolbox` aparenta estar desalinhada ou menos consistente:

- odometria: erro medio `1.243 m`, erro final `5.108 m`
- EKF + scan matching: erro medio `1.043 m`, erro final `4.845 m`

As imagens de diagnostico estao em:

- `results/sala2/slam_only_on_map_diagnostic.png`
- `results/sala2/slam_and_raw_odom_diagnostic.png`

## Como correr o corredor

Criar ambiente:

```bash
cd "/Users/diogocosta/Documents/New project 2/versão final"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Correr o EKF final:

```bash
python scripts/ekf_correlative_scanmatch.py \
  --odom data/corredor/odometry.csv \
  --slam data/corredor/slam_pose.csv \
  --lidar data/corredor/lidar_yaw_minus90.npz \
  --map-image maps/corredor/corredor_slam_map.pgm \
  --map-yaml maps/corredor/corredor_slam_map.yaml \
  --output-dir results/corredor_recomputed \
  --scan-every 5 \
  --beam-stride 5 \
  --max-points 80 \
  --xy-window 0.35 \
  --xy-step 0.10 \
  --theta-window 0.18 \
  --theta-step 0.06 \
  --score-gate 0.12 \
  --trim 0.7 \
  --std-xy 0.25 \
  --std-theta 0.10
```

## Frase curta para defesa

O EKF e o algoritmo principal: a odometria faz a predicao do movimento e o LiDAR corrige essa predicao atraves de scan matching contra um mapa gerado pela `slam_toolbox`. A pose da `slam_toolbox` nao e usada para corrigir o EKF, apenas para avaliar o erro final.
