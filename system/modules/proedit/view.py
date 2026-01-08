import customtkinter as ctk
import os
import webbrowser
from pathlib import Path

from modules.base import StudioModule


class ProEditModule(StudioModule):
    def __init__(self, parent):
        super().__init__(parent, "proedit", "ProEdit")
        self.view = None
        self.app = parent

    def get_view(self) -> ctk.CTkFrame:
        self.view = ProEditView(self.app.main_container, self.app)
        return self.view

    def on_enter(self):
        pass

    def on_leave(self):
        pass


class ProEditView(ctk.CTkFrame):
    def __init__(self, master, app, **kwargs):
        super().__init__(master, **kwargs)
        self.app = app
        self.tr = app.tr

        self.app_root = Path(__file__).resolve().parents[4]
        self.backend_dir = self.app_root / "system" / "ai-backends" / "ProEdit"
        self.output_dir = self.app_root / "system" / "proedit-out"

        self.build_ui()
        self.refresh_buttons()

    def build_ui(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(10, 5))

        ctk.CTkLabel(header, text=self.tr("proedit_title"), font=ctk.CTkFont(size=24, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(header, text=self.tr("proedit_subtitle"), text_color="gray").pack(anchor="w")

        notice = ctk.CTkFrame(self)
        notice.pack(fill="x", padx=10, pady=(20, 10))
        ctk.CTkLabel(notice, text=self.tr("proedit_coming_soon"), font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=10, pady=(10, 0))
        ctk.CTkLabel(notice, text=self.tr("proedit_coming_soon_desc"), text_color="gray").pack(anchor="w", padx=10, pady=(0, 10))

        links = ctk.CTkFrame(self, fg_color="transparent")
        links.pack(fill="x", padx=10, pady=(5, 10))

        ctk.CTkButton(links, text=self.tr("proedit_btn_open_repo"), command=self.open_repo).pack(side="left", padx=5)
        ctk.CTkButton(links, text=self.tr("proedit_btn_open_page"), command=self.open_page).pack(side="left", padx=5)

    def refresh_buttons(self):
        pass

    def open_backend_folder(self):
        if self.backend_dir.exists():
            os.startfile(str(self.backend_dir))

    def open_repo(self):
        webbrowser.open("https://github.com/iSEE-Laboratory/ProEdit")

    def open_page(self):
        webbrowser.open("https://isee-laboratory.github.io/ProEdit/")

    def open_output_folder(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        os.startfile(str(self.output_dir))
