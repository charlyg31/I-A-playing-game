
# IA Playing Game - Yu-Gi-Oh! Forbidden Memories

Ce projet est une intelligence artificielle avancée capable de jouer automatiquement à **Yu-Gi-Oh! Forbidden Memories** sur PlayStation via l’émulateur ePSXe. L’IA observe l’écran du jeu, entend les sons, interagit avec les menus, apprend par essais/erreurs, et améliore ses stratégies au fil des parties.

---

## Fonctionnalités clés

- **Vision par ordinateur hybride** :
  - Analyse l’écran original + plusieurs versions pré-traitées (grayscale, contraste, seuils, etc.).
  - OCR optimisé ligne par ligne + post-traitement contextuel.
  - Détection automatique des curseurs, menus, cartes, zones interactives.

- **IA décisionnelle** :
  - Apprend les mécaniques du jeu sans aide extérieure.
  - Gère le deck, les fusions, les styles d’adversaires, et les alignements.
  - S’adapte à l’évolution du jeu au fil des matchs.

- **Mémoire persistante** :
  - Garde en mémoire les cartes, decks ennemis, stratégies efficaces, erreurs passées.

- **Interface graphique (GUI)** :
  - Boutons : Start, Pause, Stop.
  - Indicateur visuel des touches pressées.
  - Zone de log IA et messages vocaux.

- **Voix naturelle** :
  - Réponses intelligentes orientées jeu.
  - Interaction vocale possible avec les viewers Twitch via le chat.

- **Audio** :
  - Détection précise des sons du jeu via la fenêtre ePSXe uniquement.
  - Compréhension de feedback audio (carte ajoutée, erreur, deck plein...).

- **Modules adaptatifs** :
  - L’IA peut créer dynamiquement ses propres modules internes lorsqu’elle découvre de nouvelles mécaniques (ex: alignements, modification de deck, score...).

---

## Lancement

### Pré-requis

- Python 3.11+
- Windows
- Émulateur **ePSXe** configuré
- `pyvjoy` pour le contrôle de la manette virtuelle
- `Tesseract` OCR installé

### Démarrage

```bash
# Activez l’environnement virtuel
venv\Scripts\activate

# Lancez l’IA
python run.py
```

---

## Contribution

Ce projet évolue constamment. Si tu veux suggérer une amélioration ou corriger un bug, tu peux créer une **Issue** ou proposer une **Pull Request**.

---

## Auteur

Développé par **charlyg31** avec l’assistance de GPT. Projet passionné autour de l’IA et du rétro-gaming.

---

## Licence

Ce projet est open-source et disponible sous licence MIT.
