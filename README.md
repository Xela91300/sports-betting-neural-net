# Sports Betting Neural Networks – Football, Tennis, Basketball

Projet open-source pour créer et entraîner des **réseaux de neurones** simples (Keras) afin de prédire les issues de matchs dans 3 sports :

- Football (soccer) → victoire domicile
- Tennis → victoire joueur 1
- Basketball (NBA style) → victoire domicile + over/under optionnel

**Objectif** : Fournir un squelette modulaire, propre et extensible. Pas de garantie de profit – c'est éducatif / expérimental.

## Fonctionnalités

- Modèles MLP basiques (multi-layer perceptron)
- Préparation de données sport-spécifique
- Entraînement / validation / évaluation
- Prédiction sur nouveaux matchs
- Backtest basique vs cotes (expected value)

## Installation

```bash
git clone https://github.com/ton-pseudo/sports-betting-neural-net.git
cd sports-betting-neural-net
pip install -r requirements.txt
