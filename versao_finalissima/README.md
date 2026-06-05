# Versao Finalissima - EKF Localization Pioneer 3-DX

Esta pasta contem a versao finalissima do projeto. O metodo principal continua igual a versao final: EKF com predicao por odometria e correcao por LiDAR scan matching contra um mapa gerado por `slam_toolbox`. A diferenca e que esta versao acrescenta metricas mais completas e validacao extra para reforcar o relatorio e a discussao.

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
- `scripts/metrics_nees_nis.py`: calcula metricas adicionais de trajetoria, incluindo ATE, RPE, heading error e plots de erro.
- `maps/corredor`: mapa criado pela `slam_toolbox` para o corredor.
- `maps/sala2`: mapa criado pela `slam_toolbox` para a sala.
- `data/corredor`: odometria, LiDAR e pose SLAM do corredor.
- `data/sala2`: odometria, LiDAR e pose SLAM da sala. A pose SLAM da sala e considerada incerta.
- `results/corredor`: resultado principal, com boa ground truth.
- `results/sala2`: resultado secundario e diagnostico de ground truth possivelmente desalinhada.
- `results/corredor/metrics`: metricas adicionais do corredor para o relatorio finalissimo.
- `validation_extra/simulation`: micro-simulador e resultados sinteticos.
- `validation_extra/ekf_v2_experimental`: EKF v2 experimental com Mahalanobis/NIS. Nao substitui o metodo principal.
- `report_patches`: texto pronto para adaptar ao relatorio.

## Resultado principal

O corredor e o caso usado como resultado principal:

- odometria: erro medio `0.986 m`, erro final `1.325 m`
- EKF + scan matching: erro medio `0.179 m`, erro final `0.097 m`
- EKF + scan matching: ATE-RMSE `0.220 m`, ATE-max `0.537 m`, heading-RMSE `3.45 deg`

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

## Validacao extra

### Metricas adicionais do corredor

| Metodo | ATE mean (m) | ATE RMSE (m) | Final (m) | ATE max (m) | Heading RMSE (deg) | RPE RMSE (m) |
|---|---:|---:|---:|---:|---:|---:|
| Odometry | 0.986 | 1.048 | 1.325 | 1.710 | 4.15 | 0.0534 |
| EKF final | 0.179 | 0.220 | 0.097 | 0.537 | 3.45 | 0.0528 |
| EKF v2 experimental | 0.193 | 0.229 | 0.090 | 0.581 | 3.35 | 0.0538 |

O EKF v2 experimental tem Mahalanobis gating e NIS, mas nao substitui o EKF final porque o RMSE fica ligeiramente pior. Ele e mantido como analise tecnica adicional.

### Micro-simulador

O micro-simulador valida a pipeline em condicoes com ground truth sintetico perfeito. O resultado mais forte e o caso `loop/high noise`, onde a odometria tem RMSE `1.342 m` e o EKF reduz para `0.091 m`. Tambem ha casos em que o EKF nao melhora, como `lshape/medium`, mostrando que a qualidade da correcao depende da geometria e da informatividade do scan.

### Kidnapping

O teste sintetico de kidnapping mostra que o EKF consegue detetar inconsistencia atraves de um aumento da inovacao/NIS, mas nao resolve relocalizacao global sozinho. Isto fica como limitacao e trabalho futuro.

## Como correr o corredor

Criar ambiente:

```bash
cd "/Users/diogocosta/Documents/New project 2/versão final"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Calcular metricas adicionais:

```bash
python scripts/metrics_nees_nis.py \
  --estimate results/corredor/ekf_estimate.csv \
  --reference data/corredor/slam_pose.csv \
  --label corredor_ekf_v1 \
  --output-dir results/corredor/metrics_recomputed
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
