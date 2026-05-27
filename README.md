# Pioneer 3-DX EKF Localization

Projeto de Sistemas Autonomos para localizacao de um Pioneer 3-DX num mapa pre-existente usando odometria, LiDAR e Extended Kalman Filter (EKF). O projeto foi desenvolvido para permitir trabalhar os dados offline em Python, mesmo num computador sem ROS 2 instalado.

## Objetivo

O objetivo principal e estimar a pose do robo, isto e, `x`, `y` e `theta`, combinando:

- odometria das rodas, usada para prever o movimento do robo;
- LiDAR, usado para observar paredes/landmarks do mapa e corrigir o drift;
- EKF, usado como algoritmo principal de localizacao;
- Particle Filter, usado experimentalmente como metodo de relocalizacao global quando a pose inicial e desconhecida ou em situacoes de kidnapping.

## Estrutura Do Repositorio

- `project_imp/scripts/` - scripts Python principais do projeto.
- `project_imp/maps/` - mapas usados no projeto, incluindo o mapa real principal e o mapa de simulacao.
- `project_imp/landmarks/` - landmarks extraidos dos mapas.
- `project_imp/routes/` - rotas planeadas para as simulacoes.
- `project_imp/results/` - resultados das simulacoes com varios niveis de ruido.
- `project_imp/real_bag_main/` - versao principal da bag real do Pioneer, ja calibrada.
- `project_imp/images/` - imagens escolhidas para apresentacao/relatorio.
- `project_imp/docs/` - explicacoes detalhadas de partes importantes do codigo.
- `HOW_TO_RUN.md` - guia pratico para criar ambiente Python e correr os scripts.
- `requirements.txt` - dependencias Python minimas.

As experiencias antigas e ficheiros pesados de exploracao foram deixados fora do Git atraves do `.gitignore`, para o repositorio ficar mais simples de clonar e ler.

## Pipeline Geral

1. Converter uma ROS 2 bag para formatos simples (`CSV` para odometria e `NPZ` para LiDAR).
2. Gerar mapas de ocupacao a partir de odometria + LiDAR.
3. Limpar mapas e extrair landmarks lineares.
4. Planear uma rota segura com A*.
5. Simular odometria e LiDAR com ruido.
6. Aplicar EKF usando odometria na predicao e LiDAR/landmarks na correcao.
7. Reconstruir o mapa usando as poses corrigidas pelo EKF.
8. Testar relocalizacao global com Particle Filter.
9. Aplicar a pipeline calibrada a uma bag real do Pioneer.

## Resultado Principal Da Bag Real

A versao final assumida como principal esta em `project_imp/real_bag_main/`.

Calibracao usada:

- escala em `x`: `0.590`
- escala em `y`: `0.642`
- rotacao inicial do mapa/odometria: `-40 deg`
- offset angular do LiDAR: `-90 deg`

Ficheiros importantes:

- `project_imp/real_bag_main/map/main_real_map.png`
- `project_imp/real_bag_main/landmarks/main_real_landmarks.json`
- `project_imp/real_bag_main/results/ekf_overlay.png`
- `project_imp/real_bag_main/results/mapping_ekf.png`
- `project_imp/real_bag_main/results/mapping_odom_vs_ekf.png`

Nota importante: na bag real nao existe ground truth. Por isso, os resultados reais devem ser avaliados por consistencia visual, alinhamento com o mapa e comparacao com a odometria alinhada, nao como erro absoluto real.

## Scripts Principais

- `rosbag_to_simple.py` - converte uma ROS 2 bag para `odometry.csv` e `lidar_scans.npz`.
- `bag_to_map.py` - cria um mapa de ocupacao a partir de odometria + LiDAR.
- `bag_to_hit_map.py` - cria um mapa de hits do LiDAR para analisar paredes, ruido e fronteiras.
- `clean_occupancy_map.py` - limpa mapas de ocupacao removendo ruido pequeno.
- `extract_line_landmarks.py` - extrai landmarks lineares a partir de paredes/linhas do mapa.
- `plan_generic_route.py` - planeia uma rota livre de colisoes com A*.
- `simulate_noisy_sensors.py` - gera odometria e LiDAR simulados com ruido.
- `sim_sensors_to_map.py` - reconstrui mapas a partir dos sensores simulados ou estimados.
- `ekf_localization.py` - implementacao principal do EKF.
- `particle_filter_relocalization_test.py` - teste de relocalizacao global em simulacao.
- `hybrid_pf_to_ekf_test.py` - arquitetura Particle Filter -> EKF.
- `particle_filter_real_bag.py` - teste de relocalizacao global na bag real.

## Como Correr

Ver `HOW_TO_RUN.md`.

## Permissoes

Este repositorio e publico para consulta e clone. Sem permissao de colaborador no GitHub, outras pessoas conseguem ler/clonar, mas nao conseguem fazer push para este repositorio.

