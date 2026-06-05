# Patches para adaptar o relatorio a versao finalissima

## 1. Atualizar tabela dos resultados reais

Substituir a tabela simples de erro medio/final por:

```latex
\begin{table}[H]
\centering
\caption{Localization performance in the corridor experiment using the \texttt{slam\_toolbox} reference trajectory only for evaluation.}
\begin{tabular}{lccccc}
\hline
Method & ATE Mean & ATE RMSE & Final & ATE Max & Heading RMSE \\
 & (m) & (m) & (m) & (m) & (deg) \\
\hline
Odometry & 0.986 & 1.048 & 1.325 & 1.710 & 4.15 \\
EKF + Scan Matching & 0.179 & 0.220 & 0.097 & 0.537 & 3.45 \\
EKF v2 Experimental & 0.193 & 0.229 & 0.090 & 0.581 & 3.35 \\
\hline
\end{tabular}
\end{table}
```

Texto recomendado depois da tabela:

```latex
The final EKF reduced the ATE-RMSE from $1.048$ m to $0.220$ m and the final error from $1.325$ m to $0.097$ m. An additional EKF v2 experiment with Mahalanobis gating and NIS logging produced similar accuracy, with slightly better final error but slightly worse RMSE. Therefore, the original EKF scan-matching pipeline was kept as the final method, while the v2 result is used as a consistency analysis.
```

## 2. Acrescentar paragrafo curto sobre EKF v2 / NIS

```latex
As an additional consistency check, a second EKF version was tested using Mahalanobis gating on the scan-matching innovation. The Normalised Innovation Squared (NIS) was logged for each accepted scan-matching update. In the corridor experiment, this version rejected 66 out of 1435 scan-matching candidates and obtained a mean NIS of approximately 1.98. This indicates that the filter was not over-confident, although the final RMSE was slightly worse than the main EKF version.
```

## 3. Atualizar Simulation Stage

```latex
The simulation stage was used as a controlled validation step, since the true trajectory is known by construction. The most significant improvement was observed in the loop trajectory with high odometry noise, where the odometry RMSE was $1.342$ m and the EKF reduced it to $0.091$ m. However, not all simulated cases improved: in the L-shaped trajectory with medium noise, odometry was already accurate and the EKF correction slightly increased the RMSE. This shows that scan-matching corrections are most useful when odometry drift is significant and the map geometry provides enough constraints.
```

## 4. Kidnapping / relocalizacao

```latex
A synthetic kidnapping test was also performed in simulation. The experiment showed that a sudden pose displacement causes a clear inconsistency in the scan-matching innovation, which can be detected through the NIS. However, the EKF is a single-hypothesis Gaussian filter and cannot recover from a large global displacement without an additional global relocalization module. For this reason, particle-filter or AMCL-based relocalization is left as future work.
```

## 5. Frase importante para evitar erro conceptual

Usar sempre:

```latex
The \texttt{slam\_toolbox} trajectory is used as an evaluation reference, not as an input to the EKF.
```

Evitar:

```latex
ground truth from slam_toolbox
```

