from abc import ABC, abstractmethod
import customtkinter as ctk

class StudioModule(ABC):
    def __init__(self, parent, name, title):
        self.parent = parent
        self.name = name
        self.title = title

    @abstractmethod
    def get_view(self) -> ctk.CTkFrame:
        """Returns the main frame of the module to be displayed."""
        pass

    @abstractmethod
    def on_enter(self):
        """Called when the module becomes active."""
        pass

    @abstractmethod
    def on_leave(self):
        """Called when navigating away from the module."""
        pass
