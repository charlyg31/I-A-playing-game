
import os
import json
from datetime import datetime

class AutoDiscoverManager:
    def __init__(self, base_path="core/discovery"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        self.known_mechanics = {}

    def discover_mechanic(self, name_hint, context, result):
        module_name = f"{name_hint.lower().replace(' ', '_')}_tracker"
        file_path = os.path.join(self.base_path, f"{module_name}.py")

        if not os.path.exists(file_path):
            self._create_new_tracker_module(file_path, module_name)
            print(f"[AutoDiscover] Nouveau module découvert : {module_name}")

        data_file = file_path.replace(".py", ".json")
        entry = {"context": context, "result": result, "timestamp": datetime.now().isoformat()}

        if os.path.exists(data_file):
            with open(data_file, "r") as f:
                data = json.load(f)
        else:
            data = []

        data.append(entry)
        with open(data_file, "w") as f:
            json.dump(data, f, indent=2)

    def _create_new_tracker_module(self, file_path, class_name):
        template = f"""class {class_name.title().replace('_', '')}:
    def __init__(self):
        self.observations = []

    def observe(self, context, result):
        self.observations.append({{"context": context, "result": result}})
"""
        with open(file_path, "w") as f:
            f.write(template)
