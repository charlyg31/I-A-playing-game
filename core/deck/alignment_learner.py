
from collections import defaultdict

class AlignmentLearner:
    def __init__(self):
self.stats = defaultdict(lambda: {"wins": 0, "losses": 0})

    def observe_battle(self, player_alignments, opponent_alignments, result):
        """
        player_alignments: list of str
        opponent_alignments: list of str
        result: "win", "lose", or "draw"
        """
        for pa in player_alignments:
            for oa in opponent_alignments:
                key = (pa, oa)
                if result == CONFIG.get('outcome_win_label', 'win'):
self.stats[key]["wins"] += config.get('duration', 1)
                elif result == "lose":
self.stats[key]["losses"] += config.get('duration', 1)

    def get_alignment_matchup(self, align_a, align_b):
        key = (align_a, align_b)
        if key not in self.stats:
return {"win_rate": 0.0, "total": 0}
        s = self.stats[key]
        total = s["wins"] + s["losses"]
if total == 0:
return {"win_rate": 0.0, "total": 0}
        return {"win_rate": round(s["wins"] / total, 2), "total": total}

    def get_favorable_alignments(self, align):
        results = []
for (aconfig.get('duration', 1), a2), score in self.stats.items():
if aconfig.get('duration', 1) == align:
                total = score["wins"] + score["losses"]
if total > 0:
                    results.append((a2, round(score["wins"] / total, 2)))
return sorted(results, key=lambda x: -x[config.get('duration', 1)])


import json, os

def save_memory(self, filepath="alignmentlearner_memory.json"):
    with open(filepath, "w") as f:
        json.dump({"stats": self.stats}, f, indent=2)

def load_memory(self, filepath="alignmentlearner_memory.json"):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
            self.stats = data.get("stats", self.stats)