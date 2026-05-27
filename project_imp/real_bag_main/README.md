# Real Pioneer Bag Main Version

Esta pasta contem a versao principal da bag real do Pioneer.

Parametros assumidos:

- `scale_x = 0.590`
- `scale_y = 0.642`
- rotacao inicial do mapa/odometria: `-40 deg`
- offset angular do LiDAR: `-90 deg`
- range maximo do LiDAR: `5.6 m`

Ficheiros principais:

- `map/main_real_map.*`: mapa principal retangular.
- `landmarks/main_real_landmarks.json`: landmarks principais do mapa.
- `data/raw_odometry.csv`: odometria original convertida da bag.
- `data/aligned_odometry.csv`: odometria alinhada ao mapa principal.
- `data/lidar_yaw_minus90.npz`: LiDAR corrigido com offset de `-90 deg`.
- `results/ekf_estimate.csv`: trajetoria estimada pelo EKF.
- `results/ekf_overlay.png`: comparacao odometria/EKF.
- `results/mapping_ekf.png`: mapa reconstruido usando a trajetoria EKF.

Experiencias antigas e hipoteses rejeitadas foram arquivadas em:

- `project_imp/archive/real_bag_experiments/`
