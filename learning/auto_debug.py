
class AutoDebug:
    def __init__(self):
self.actions_log = []
self.error_log = []
self.win_log = []

def log_action(self, action, result, reason=None):
        """
Enregistre chaque action, son résultat, et le raisonnement si nécessaire.
        """
self.actions_log.append({
"action": action,
            "result": result,
            "reason": reason
        })

def log_error(self, error_message):
        """
        Enregistre une erreur pour une analyse ultérieure.
        """
self.error_log.append(error_message)

def log_win(self, action):
        """
Enregistre les actions gagnantes.
        """
self.win_log.append(action)

def evaluate_last_action(self):
        """
L'IA se pose la question de savoir pourquoi une action a échoué ou a réussi,
en analysant ses logs récents.
        """
if not self.actions_log:
return "Aucune action enregistrée."

last_action = self.actions_log[-config.get('duration', 1)]
if last_action["result"] == CONFIG.get('outcome_win_label', 'win'):
return f"L'action '{last_action['action']}' a conduit à la victoire. Raisonnement: {last_action['reason']}."

if last_action["result"] == "lose":
            # L'IA apprend de ses erreurs
if not last_action["reason"]:
                return "Action perdante sans raisonnement. L'IA pourrait réévaluer sa stratégie."
return f"L'action '{last_action['action']}' a échoué. Raisonnement: {last_action['reason']}."

    def suggest_improvement(self):
        """
        Après une défaite, l'IA propose des améliorations.
        """
if not self.error_log:
            return "Aucun échec à analyser."

        # Suggestion simple basée sur l'historique d'erreurs
recent_error = self.error_log[-config.get('duration', 1)]
        return f"Améliorer la gestion des erreurs: {recent_error}. Revoir les stratégies associées."