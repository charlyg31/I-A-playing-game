from core.opponent.adversary_predictor import AdversaryPredictor
from core.opponent.opponent_memory import OpponentMemory
from core.session.name_input_detector import NameInputDetector
from core.session.session_namer import SessionNamer
from core.opponent.adversary_deck_memory import AdversaryDeckMemory
from core.opponent.enemy_tracker import EnemyTracker
from core.strategy.victory_predictor import VictoryPredictor
from core.strategy.combo_learner import ComboLearner
from core.deck.deck_memory import DeckMemory
from core.deck.card_evaluator import CardEvaluator
from core.deck.deck_mutator import DeckMutator
from core.strategy.goal_manager import GoalManager
from game_analysis.audio_reasoning import AudioReasoner
from learning.deck_brain import DeckBrain
from chat_interaction.context_speaker import ContextSpeaker

import time
import cv2
import numpy as np
import easyocr
from core.vision.game_analyzer import GameAnalyzer
from learning.qlearning_agent import QLearningAgent
from learning.auto_reward import AutoRewardEngine
from learning.strategy_memory import StrategyMemory
from game_analysis.logger_system import LoggerSystem
from input_simulation.input_simulator import InputSimulator
from interface.tts_responder import TTSResponder
from PIL import Image

# Initialisation des modules IA
analyzer = GameAnalyzer()
agent = QLearningAgent()
reward_engine = AutoRewardEngine()
memory = StrategyMemory()
logger = LoggerSystem()
simulator = InputSimulator()
tts = TTSResponder()
deck_brain = DeckBrain()
goal_manager = GoalManager()
name_input_detector = NameInputDetector()
session_namer = SessionNamer()
adversary_memory = AdversaryDeckMemory()
opponent_memory = OpponentMemory()
adversary_predictor = AdversaryPredictor()
current_opponent = None
enemy_tracker = EnemyTracker()
victory_predictor = VictoryPredictor()
combo_learner = ComboLearner()
combo_history = []
deck_memory = DeckMemory()
card_evaluator = CardEvaluator()
deck_mutator = DeckMutator(all_cards=['Baby Dragon', 'Thunder Kid', 'Skull Red Bird', 'Fire Grass', 'Mystical Elf'])
current_deck = deck_mutator.mutate_deck(['Baby Dragon'] * 40)
audio_reasoner = AudioReasoner()
context_speaker = ContextSpeaker()
last_texts = []
reader = easyocr.Reader(['en'], gpu=False)

available_actions = simulator.get_available_actions()
previous_state = None
prev_state_key = None
last_action = None
seen_texts = set()

def build_state_key(state):
    context = state.get("context", "")
    hand = "|".join(state.get("hand", []))
    return f"{context}|{hand}"

def extract_lines_easyocr(image):
    raw = np.array(image)
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(equalized)
    resized = cv2.resize(enhanced, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    results = reader.readtext(resized)
    lines = {}

    for bbox, text, conf in results:
        if len(text.strip()) <= 1:
            continue
        y_center = int(sum([point[1] for point in bbox]) / 4)
        found_line = None
        for key in lines:
            if abs(key - y_center) <= 15:
                found_line = key
                break
        if found_line is None:
            lines[y_center] = []
            found_line = y_center
        x_coord = int(min(p[0] for p in bbox))
        lines[found_line].append((x_coord, text.strip()))

    final_lines = []
    for y in sorted(lines.keys()):
        line = " ".join([t for x, t in sorted(lines[y], key=lambda x: x[0])])
        final_lines.append(line)
    return final_lines

def interpret_situation(context_label, hand, tags):
    if context_label == "duel_screen" and hand:
        return "Je suis en plein duel et je vois des cartes prêtes à être jouées."
    elif context_label == "deck_editor":
        return "Je suis dans l'éditeur de deck."
    elif context_label == "menu_or_dialogue":
        return "Je suis dans un menu ou une scène de dialogue."
    elif context_label == "navigation":
        return "Je suis en phase de navigation entre les écrans du jeu."
    else:
        return "Je vois une interface, j'observe pour mieux comprendre."

while True:
    screenshot = analyzer.capture_screen()
    if screenshot is None:
        continue

    vis = np.array(screenshot).copy()
    current_state = analyzer.get_game_state(screenshot)
    state_key = build_state_key(current_state)

    context_label = current_state.get("context_label", current_state.get("context"))
    hand = current_state.get("hand", [])
    visual_tags = current_state.get("visual_tags", [])
    interpretation = interpret_situation(context_label, hand, visual_tags)

    print(f"[IA] {interpretation}")

    # Lire lignes complètes
    detected_lines = extract_lines_easyocr(screenshot)
    
    for line in detected_lines:
        if len(line.strip()) <= 3 or len(line.split()) == 1:
            continue  # ignorer les mots courts isolés

        message = f"Je vois une ligne contenant : {line}"
        if "SAVE" in line.upper():
            message += " — Je pense que c'est une option pour sauvegarder."
        elif "DECK" in line.upper():
            message += " — Cela concerne sûrement la gestion de mon deck."
        elif "TITLE" in line.upper():
            message += " — Cela semble me ramener au menu principal."
        elif "LEAVE" in line.upper() or "SHOP" in line.upper():
            message += " — Je crois que cela me permet de quitter la boutique."

        if line not in seen_texts:
            seen_texts.add(line)
            print(f"[IA] {message}")
            try:
                tts.speak(message)
            except Exception as e:
                print(f"[ERREUR TTS] {e}")

        if line not in seen_texts:
            seen_texts.add(line)
            print(f"[IA] Nouvelle ligne détectée : {line}")
            try:
                tts.speak(f"Je vois une ligne contenant : {line}")
            except Exception as e:
                print(f"[ERREUR TTS] {e}")

    if previous_state is not None and last_action is not None:
        reward = reward_engine.evaluate(previous_state, last_action, current_state)
        agent.update_q_table(prev_state_key, last_action, reward, state_key)
        logger.log(current_state, last_action, reward)
        print(f"[IA] Récompense reçue pour la touche {last_action} : {reward}")

    strategy_action = memory.get_best_sequence(state_key)
    if strategy_action:
        action = strategy_action
        reason = "car j'ai appris que cette touche est efficace ici"
    else:
        action = agent.choose_action(state_key, available_actions)
        reason = "pour explorer cette situation et voir ce qu'elle déclenche"

    vocal_message = f"{interpretation} Je vais appuyer sur la touche {action} {reason}."
    print(f"[IA] {vocal_message}")
    try:
        tts.speak(vocal_message)
    except Exception as e:
        print(f"[ERREUR TTS vocal_message] {e}")

    simulator.perform(action)
    logger.log_event(f"Touche appuyée (vJoy) : {action}")

    # Sauvegarde image vision
    cv2.imwrite("debug_frame.png", vis)

    # Mode headless : pas d'affichage, seulement sauvegarde d'image
    # Vous pouvez ouvrir debug_frame.png manuellement

    last_action = action
    previous_state = current_state
    prev_state_key = state_key
