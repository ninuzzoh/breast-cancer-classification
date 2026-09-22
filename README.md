# Classificazione supervisionata - Breast Cancer Wisconsin

Confronto di quattro classificatori per la diagnosi di masse mammarie benigne e maligne
a partire da caratteristiche morfologiche del nucleo cellulare (dataset WDBC, UCI).

## Obiettivo

Il problema è trattato come classificazione binaria supervisionata con un'asimmetria
dei costi esplicita: mancare un tumore maligno (falso negativo) è clinicamente molto
più grave che richiamare una paziente sana. Tutta la pipeline - dalla scelta della
metrica di tuning all'analisi della soglia decisionale - è costruita attorno a questo
vincolo.

## Modelli confrontati

- Logistic Regression
- Linear Discriminant Analysis (LDA)
- Support Vector Machine con kernel RBF
- Random Forest

## Risultati sul test set

| Modello              | Accuracy | Recall | F1    | AUC   |
|----------------------|----------|--------|-------|-------|
| Logistic Regression  | 0.973    | 1.000  | 0.965 | 0.996 |
| LDA                  | 0.965    | 0.905  | 0.950 | 0.998 |
| SVM (RBF)            | 0.974    | 0.929  | 0.963 | 0.993 |
| Random Forest        | 0.974    | 0.929  | 0.963 | 0.995 |

La SVM-RBF è il modello più affidabile: recall medio più alto in cross validation
ripetuta (0.966) e deviazione standard più contenuta (0.021).

## Aspetti principali

- **Prevenzione del data leakage**: ogni modello è incapsulato in una Pipeline
  con StandardScaler, i parametri di scaling vengono stimati solo sul training set
- **Tuning con GridSearchCV** ottimizzando il recall su cross validation stratificata
- **Analisi della soglia decisionale**: abbassando la soglia della LDA a 0.044
  si eliminano tutti i falsi negativi al costo di 5 falsi positivi
- **Interpretabilità e spiegabilità**: coefficienti logistici, feature importance
  e valori SHAP sulla Random Forest

## Dataset

Breast Cancer Wisconsin (Diagnostic) - caricato automaticamente via `ucimlrepo`,
569 osservazioni, 30 feature numeriche, nessun valore mancante.

## Esecuzione

```bash
pip install -r requirements.txt
jupyter notebook breast_cancer_classification.ipynb
```

## Tecnologie

Python · scikit-learn · SHAP · pandas · matplotlib · seaborn · ucimlrepo
