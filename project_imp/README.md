# Project Important Files

Esta pasta junta os ficheiros mais importantes do projeto para explicar o trabalho e responder a perguntas sobre o código.

## Estrutura

- `scripts/` - scripts Python principais usados no pipeline.
- `maps/` - mapa final de simulação em PNG/PGM/YAML.
- `landmarks/` - landmarks extraídos do mapa final.
- `routes/` - rota planeada para o robô.
- `results/` - resultados de simulação, EKF e mapping para vários perfis de ruído.
- `real_bag_main/` - versão principal da bag real do Pioneer, já calibrada.
- `images/` - imagens escolhidas para a apresentação.
- `data/` - dados simplificados das bags reais.
- `docs/` - notas explicativas.

## Pipeline geral

1. Converter/ler dados das bags reais.
2. Criar mapas por LiDAR usando odometria.
3. Limpar mapas e extrair landmarks de linhas.
4. Criar planta simplificada para simulação.
5. Planear rota segura.
6. Simular odometria com ruído e LiDAR.
7. Aplicar EKF usando landmarks de linhas.
8. Reconstruir o mapa usando as poses corrigidas pelo EKF.
9. Comparar resultados entre `starter`, `stable`, `strong` e `very_strong`.

## Resultado principal no mapa final

O EKF usa 16 landmarks lineares extraídos das fronteiras do mapa.

- `starter`: EKF final perto de 1 cm.
- `stable`: EKF final perto de 1 cm.
- `strong`: EKF final perto de 5 cm.
- `very_strong`: EKF final abaixo de 10 cm na configuração mais permissiva.

O ponto importante para defender: a odometria degrada com ruído acumulado, mas o EKF consegue corrigir a pose quando observa paredes/linhas conhecidas no mapa.

## Versão Principal Da Bag Real

A versão assumida como correta para a bag real está em `real_bag_main/`.

Calibração usada:

- `scale_x = 0.590`
- `scale_y = 0.642`
- rotação inicial do mapa/odometria: `-40 deg`
- offset angular do LiDAR: `-90 deg`

Ficheiros principais também têm aliases simples:

- `maps/main_real_map.*`
- `landmarks/main_real_landmarks.json`
- `real_bag_main/data/aligned_odometry.csv`
- `real_bag_main/data/lidar_yaw_minus90.npz`
- `real_bag_main/results/ekf_overlay.png`
- `real_bag_main/results/mapping_ekf.png`

As tentativas antigas de escala, rotação, yaw do LiDAR e ICP foram arquivadas em `archive/real_bag_experiments/`.
