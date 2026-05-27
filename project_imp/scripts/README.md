# Scripts principais

Estes sao os scripts que ficam como parte importante do projeto.

## Bags e mapas

- `rosbag_to_simple.py`: converte dados extraidos das bags para CSV/NPZ simples.
- `bag_to_map.py`: cria mapa de ocupacao a partir de odometria + LiDAR.
- `bag_to_hit_map.py`: cria mapa de hits do LiDAR para analisar ruido e fronteiras.
- `clean_occupancy_map.py`: limpa mapas de ocupacao, removendo ruido pequeno.

## Mapa simulado e sensores

- `extract_line_landmarks.py`: extrai landmarks de linhas/parede do mapa.
- `plan_generic_route.py`: planeia uma rota livre de colisoes com A*.
- `simulate_noisy_sensors.py`: gera odometria e LiDAR simulados com ruido.
- `sim_sensors_to_map.py`: reconstrucao/visualizacao de mapa a partir dos sensores simulados.

## Localizacao

- `ekf_localization.py`: localizador principal com EKF.
- `particle_filter_relocalization_test.py`: relocalizacao global por Particle Filter para recuperacao apos kidnapping.
- `hybrid_pf_to_ekf_test.py`: wrapper que testa a arquitetura PF -> EKF sem alterar o EKF original.

## Arquivados

Scripts que foram uteis durante a exploracao mas nao fazem parte da solucao final estao em:

- `project_imp/archive_scripts/`
