# Guia função a função: `ekf_localization.py`

Este documento explica todas as classes e funções do ficheiro `ekf_localization.py`, incluindo partes que não foram usadas na configuração final.

## Classes

- `Pose`: guarda uma pose 2D com tempo: `t, x, y, theta`.
- `CircleLandmark`: representa landmarks circulares: centro `(x, y)` e raio.
- `PointLandmark`: representa landmarks pontuais, por exemplo cantos.
- `LineLandmark`: representa uma parede/linha por dois pontos `(x1, y1)` e `(x2, y2)`.
- `Observation`: representa uma observação LiDAR em forma polar: distância, bearing, raio estimado, número de pontos e erro.
- `AssociationModel`: guarda um modelo ML de associação de landmarks circulares.
- `IcpConfidenceModel`: guarda um modelo ML simples para decidir se uma correção ICP é confiável.

## Funções auxiliares de ângulo e modelos ML

- `wrap_angle`: normaliza ângulos para `[-pi, pi]`.
- `load_association_model`: lê do JSON um modelo de regressão logística usado para associação de landmarks.
- `association_features`: calcula features entre uma observação circular e um landmark circular.
- `expand_association_features`: aplica expansão polinomial se o modelo ML tiver sido treinado com features quadráticas.
- `association_probability`: calcula a probabilidade de uma observação corresponder a um landmark usando o modelo ML.
- `landmark_observation_errors`: calcula erros de range, bearing e raio entre observação e landmark circular.
- `landmark_match_cost`: aplica gates e calcula custo de associação para landmark circular.
- `score_landmark_pose`: avalia quão bem uma pose explica várias observações circulares.
- `load_icp_confidence_model`: lê um modelo gradient boosting simples para validar ICP.
- `predict_icp_confidence`: calcula a confiança do ICP usando stumps do modelo.
- `icp_confidence_features`: cria features que descrevem a qualidade de uma correção ICP.

## Leitura de dados

- `read_poses`: lê CSV `t,x,y,theta`.
- `read_circle_landmarks`: lê landmarks circulares do JSON.
- `read_corner_landmarks`: lê landmarks pontuais/cantos do JSON.
- `read_line_landmarks`: lê landmarks lineares do JSON.

## Deteção de observações no LiDAR

- `fit_circle`: ajusta um círculo a pontos LiDAR por mínimos quadrados.
- `segment_scan`: separa o scan em segmentos contínuos, evitando misturar objetos diferentes.
- `detect_circle_observations`: encontra objetos circulares no LiDAR.
- `fit_line_segment`: ajusta uma reta a um segmento de pontos usando PCA.
- `line_intersection`: calcula interseção entre duas retas.
- `detect_corner_observations`: deteta cantos de 90 graus a partir de duas linhas observadas.
- `detect_line_observations`: deteta linhas/paredes no scan LiDAR e devolve `(rho, alpha, length, points, error)`.

## EKF prediction e updates

- `predict_from_odom_delta`: faz a fase de previsão do EKF usando o delta da odometria.
- `update_with_landmark`: update EKF usando landmark circular.
- `update_with_point_landmark`: update EKF usando landmark pontual/canto.
- `line_params_in_map`: converte uma linha do mapa para forma polar `(rho, alpha)`.
- `update_with_line_landmark`: update EKF usando landmark linear.

## Associação de observações a landmarks

- `associate_observation`: associa uma observação circular a um landmark circular.
- `globally_associate_observations`: faz associação global entre várias observações e landmarks circulares.
- `associate_point_observation`: associa uma observação de canto a um canto do mapa.
- `associate_line_observation`: associa uma linha observada pelo LiDAR a uma linha conhecida do mapa.

## Recuperação global

- `circle_intersection_pose_candidates`: gera poses candidatas usando pares de observações circulares.
- `maybe_global_recover`: tenta recuperar a pose quando a covariância está alta usando landmarks circulares.
- `pose_from_observation_pair`: calcula uma pose candidata a partir de um par observação-landmark.
- `maybe_global_landmark_relocalize`: tenta relocalização global usando várias associações de landmarks circulares.

## ICP

- `read_map_yaml`: lê resolução e origem do mapa.
- `DistanceMap`: cria um mapa de distâncias até obstáculos para scan-to-map ICP.
- `scan_points_for_icp`: converte um scan LiDAR em pontos 2D no referencial do robô.
- `icp_scan_to_map`: alinha pontos LiDAR ao mapa por ICP e devolve pose corrigida.
- `scan_to_map_rmse`: mede o erro médio dos pontos LiDAR relativamente ao mapa.
- `update_with_pose_measurement`: faz update EKF usando uma medição direta de pose, por exemplo vinda do ICP.

## Escrita, visualização e métricas

- `save_estimate`: guarda a trajetória estimada pelo EKF em CSV.
- `world_to_px`: converte coordenadas do mundo para pixels.
- `draw_overlay`: desenha trajetória real, odometria e EKF por cima do mapa.
- `path_errors`: calcula erro final e erro médio contra ground truth.
- `parse_args`: define os argumentos de linha de comandos.
- `main`: executa o pipeline completo do EKF.
