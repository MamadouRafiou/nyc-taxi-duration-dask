# Prédiction de la durée des trajets de taxi — inférence distribuée avec Dask

> Entraînement d'un modèle de régression sur 2,9 millions de trajets de taxi new-yorkais, puis inférence distribuée sur le mois suivant avec Dask
> INF8810 — Projet 1, UQAM

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Dask](https://img.shields.io/badge/Dask-distributed-FDA061?logo=dask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?logo=scikitlearn)

---

## Le problème

Un service de taxi veut annoncer au chauffeur la durée estimée d'une course **dès qu'elle est enregistrée**, et non après le traitement d'un lot complet. Deux contraintes en découlent :

1. Le modèle doit être entraîné sur un volume de données qui ne tient pas confortablement en mémoire d'un seul processus.
2. L'inférence doit être parallélisée et émettre ses résultats au fil de l'eau, partition par partition.

Le projet traite les deux avec **scikit-learn** pour l'entraînement et **Dask** pour l'inférence distribuée.

---

## Données

Trajets des taxis jaunes de New York, publiés par la NYC Taxi & Limousine Commission au format Parquet.

| Jeu | Usage | Volume |
|---|---|---|
| Janvier 2024 | Entraînement et test | 2 964 624 trajets bruts |
| Février 2024 | Inférence distribuée | Fichier complet, 16 partitions |

Les fichiers sont téléchargés dans le carnet avec `httpx` en flux, par blocs de 64 Ko, pour éviter de charger la réponse entière en mémoire.

### Nettoyage

Après filtrage des valeurs aberrantes, il reste **2 904 098 trajets** sur janvier, soit 98 % des données :

- durée strictement positive et inférieure à 24 heures ;
- distance strictement positive et inférieure à 200 milles ;
- suppression des lignes dont `VendorID`, `trip_distance` ou `trip_duration` est manquant.

---

## Modèle

| Élément | Choix |
|---|---|
| Algorithme | `RandomForestRegressor(n_estimators=10)` |
| Attributs | `trip_distance` (normalisé), `VendorID` (encodage one-hot) |
| Cible | `trip_duration`, en secondes |
| Répartition | 2 323 278 en entraînement / 580 820 en test |
| Métrique | MAE — erreur absolue moyenne |

**Résultat : MAE de 303,2 secondes, soit environ 5 minutes d'erreur moyenne par trajet.**

Deux points méritent d'être relevés.

**Le choix de la MAE.** Elle s'exprime directement en secondes, donc elle se lit sans conversion, et elle est moins sensible que la RMSE aux trajets extrêmes, nombreux dans ce jeu de données.

**Les limites assumées.** Le modèle n'utilise que deux attributs. L'heure de la journée, le jour de la semaine et la zone géographique sont absents alors qu'ils expliquent une part importante de la variance — un trajet de 5 km à 8 h en semaine n'a rien à voir avec le même trajet un dimanche à 3 h. Une MAE de 5 minutes est donc un point de départ honnête, pas un résultat optimisé.

**Robustesse de l'encodage.** `OneHotEncoder(handle_unknown='ignore')` est ajusté uniquement sur les données d'entraînement. Si un `VendorID` inconnu apparaît en production, l'encodeur produit un vecteur de zéros au lieu de lever une exception : la prédiction perd en qualité mais le service ne tombe pas.

---

## Inférence distribuée

Le fichier de février est découpé en 16 partitions, chacune étant un DataFrame pandas classique. `map_partitions` applique la fonction de prédiction à chaque partition en parallèle.

```python
ddf_feb = dd.read_parquet(PATH_FEB).repartition(npartitions=16)
ddf_with_preds = ddf_feb.map_partitions(predict_partition, meta=meta)
```

`send_to_taxi()` est appelée **à l'intérieur** de la fonction de prédiction, et non après un `.compute()` global. C'est le point central du projet : chaque chauffeur reçoit son estimation dès que sa partition est traitée, au lieu d'attendre que les millions de trajets du mois soient tous calculés.

### Une contrainte d'environnement, et comment elle a été contournée

L'énoncé demandait un cluster de 4 workers :

```python
Client(n_workers=4, threads_per_worker=1, processes=True)
```

Google Colab s'exécute dans un bac à sable qui interdit la création de sous-processus indépendants. Avec `processes=True`, Dask tente de lancer 4 processus fils par `fork` ou `spawn`, ce que Colab bloque — le noyau s'arrête avant même le traitement des données.

La configuration retenue bascule sur des fils d'exécution plutôt que des processus :

```python
Client(n_workers=1, threads_per_worker=4, memory_limit='0', processes=False)
```

On obtient 4 unités d'exécution concurrentes, fonctionnellement équivalentes à 4 workers pour `map_partitions` : les partitions sont traitées quatre par quatre. Sur une machine locale, la configuration d'origine fonctionne sans modification.

Les prédictions sont enfin écrites au format Parquet dans `predictions_february_2024/`.

---

## Exécution

```bash
pip install -r requirements.txt
jupyter notebook taxi_duration_dask.ipynb
```

Le carnet télécharge les données tout seul au premier lancement. Prévoir environ 1 Go d'espace disque pour les deux fichiers Parquet et les prédictions.

> Le carnet a été exécuté sur Google Colab. Hors Colab, remplacer la configuration du client Dask par `Client(n_workers=4, threads_per_worker=1, processes=True)`.

---

## Équipe

Projet réalisé en équipe de trois dans le cadre du cours INF8810 à l'UQAM.

- Mamadou Rafiou Barry
- Fatima Barry
- Naouel Ouazar

**Cours :** INF8810 — Systèmes distribués et ingénierie de données pour l'apprentissage automatique
**Établissement :** UQAM — Université du Québec à Montréal

---

## Source des données

[NYC Taxi & Limousine Commission — Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
