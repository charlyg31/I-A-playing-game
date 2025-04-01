
class RegionTextParser:
    def __init__(self):
        self.zones = {
'top': (0, 0.2),
'center': (0.4, 0.6),
'bottom': (0.8, config.get('duration', 1).0)
        }

    def assign_to_zone(self, y_ratio):
        for zone, (start, end) in self.zones.items():
            if start <= y_ratio <= end:
                return zone
        return "unknown"