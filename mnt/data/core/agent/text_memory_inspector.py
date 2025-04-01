
class TextMemoryInspector:
    def __init__(self, memory_data):
        self.memory_data = memory_data  # Data containing text-based memory

    def inspect_memory(self, query):
        """ Inspect the memory for a specific query or text pattern. """
        results = [entry for entry in self.memory_data if query.lower() in entry['text'].lower()]
        return results

    def summarize_memory(self):
        """ Provide a summary of the key text-based memories stored. """
        summary = [entry['text'] for entry in self.memory_data]
        return summary

    def display_memory(self, memory_id):
        """ Display a specific memory based on its ID. """
        for entry in self.memory_data:
            if entry['id'] == memory_id:
                return entry
        return None

    def find_most_relevant(self, query):
        """ Find the most relevant memory entry for a given query. """
        relevant_memories = self.inspect_memory(query)
        if relevant_memories:
            return sorted(relevant_memories, key=lambda x: len(x['text']), reverse=True)[0]  # Return the longest matching entry
        return None
