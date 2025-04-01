
import json
import time

log_data = []

def log_step(state, action, reward, next_state, turn, timestamp=None):
    if timestamp is None:
        timestamp = time.time()
    log_data.append({
        "turn": turn,
        "timestamp": timestamp,
        "state": state,
        "action": action,
        "reward": reward,
        "next_state": next_state
    })

def save_logs_to_json(filename):
    with open(filename, 'w') as f:
        json.dump(log_data, f, indent=2)
