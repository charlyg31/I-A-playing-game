
class TextMemory:
    def __init__(self):
        self.memory = []  # Liste pour stocker les mémoires textuelles

    def store_text_memory(self, text):
        """ Stocke un texte dans la mémoire. """
        self.memory.append(text)

    def retrieve_text_memory(self, index):
        """ Récupère un texte spécifique de la mémoire en fonction de l'index. """
        if 0 <= index < len(self.memory):
            return self.memory[index]
        return None

    def clear_memory(self, index):
        """ Efface un texte de la mémoire en fonction de l'index. """
        if 0 <= index < len(self.memory):
            del self.memory[index]

    def summarize_memory(self):
        """ Fournit un résumé des mémoires textuelles stockées. """
        return [text[:50] + '...' if len(text) > 50 else text for text in self.memory]  # Résumé des premiers 50 caractères
