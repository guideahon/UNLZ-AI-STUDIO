import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import os
import json
from pathlib import Path
from PIL import Image

print("DEBUG: LOADED LATEST VERSION OF STUDIO_GUI")

# Import modules
from modules.gaussian import GaussianModule
from modules.llm_frontend import LLMFrontendModule
from modules.inclu_ia import IncluIAModule

# Set Theme
THEME_PATH = os.path.join(os.path.dirname(__file__), "assets", "themes", "ingenieria.json")
if os.path.exists(THEME_PATH):
    ctk.set_default_color_theme(THEME_PATH)
ctk.set_appearance_mode("Dark")

INSTALLED_MODULES_FILE = "installed_modules.json"
LANGUAGES_FILE = os.path.join(os.path.dirname(__file__), "assets", "languages.json")

class StudioGUI(ctk.CTk):
    def __init__(self, profile_manager, preflight_report, log_dir, logo_path=None, manager=None, warm_callback=None, splash_path=None):
        super().__init__()

        # Data & Managers
        self.profile_manager = profile_manager
        self.manager = manager
        self.log_dir = log_dir
        self.installed_modules = self.load_installed()
        
        # i18n
        self.languages = self.load_languages()
        self.current_lang = "es" # Default Spanish

        # Window Setup
        self.title("UNLZ AI Studio")
        self.geometry("1280x800")
        self.minsize(1024, 768)
        
        # Grid Layout (Sidebar + Main)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Assets
        self.assets_path = os.path.join(os.path.dirname(__file__), "assets")
        
        # --- Sidebar ---
        self.sidebar_frame = ctk.CTkFrame(self, width=240, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(6, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="UNLZ AI STUDIO", font=ctk.CTkFont(size=22, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 30))

        # Nav Buttons (references kept for text update)
        self.btn_home = self.create_nav_button("nav_home", self.show_home_view, row=1)
        self.btn_modules = self.create_nav_button("nav_store", self.show_modules_view, row=2)
        self.btn_monitor = self.create_nav_button("nav_monitor", self.show_monitor_view, row=3)
        self.btn_settings = self.create_nav_button("nav_settings", self.show_settings_view, row=4)
        
        self.sidebar_separator = ctk.CTkLabel(self.sidebar_frame, text=self.tr("sidebar_installed"), font=ctk.CTkFont(size=12, weight="bold"), text_color="gray")
        self.sidebar_separator.grid(row=5, column=0, sticky="w", padx=20, pady=(20, 5))
        
        # Dynamic Module Buttons Container
        self.module_buttons_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.module_buttons_frame.grid(row=6, column=0, sticky="nsew")
        self.module_buttons_frame.grid_columnconfigure(0, weight=1)
        
        # Language Switcher & Status
        self.footer_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.footer_frame.grid(row=7, column=0, padx=20, pady=20, sticky="ew")
        
        self.lang_switcher = ctk.CTkSegmentedButton(self.footer_frame, values=["ES", "EN"], command=self.change_language)
        self.lang_switcher.set("ES")
        self.lang_switcher.pack(fill="x", pady=(0, 10))

        self.status_label = ctk.CTkLabel(self.footer_frame, text=self.tr("status_ready"), font=ctk.CTkFont(size=12), text_color="gray")
        self.status_label.pack()

        # --- Main Content Area ---
        self.main_container = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_container.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        
        # Views
        self.views = {}
        self.current_view_name = None
        self.current_view = None

        # Available Module definitions (metadata)
        self.available_modules = {
            "gaussian": {
                "class": GaussianModule,
                "title_key": "mod_gaussian_title", 
                "desc_key": "mod_gaussian_desc",
                "icon": "GS"
            },
            "llm_frontend": {
                "class": LLMFrontendModule,
                "title_key": "mod_llm_title",
                "desc_key": "mod_llm_desc",
                "icon": "AI"
            },
            "inclu_ia": {
                "class": IncluIAModule,
                "title_key": "mod_incluia_title",
                "desc_key": "mod_incluia_desc",
                "icon": "CC"
            }
        }
        
        # Loaded module instances
        self.loaded_modules = {} 
        
        # Initial Build
        self.refresh_ui()
        self.show_home_view()

    # --- i18n Helpers ---
    def load_languages(self):
        if os.path.exists(LANGUAGES_FILE):
            try:
                with open(LANGUAGES_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {}

    def tr(self, key):
        """Translate a key to the current language."""
        if self.current_lang in self.languages:
            return self.languages[self.current_lang].get(key, key)
        return key

    def change_language(self, lang):
        self.current_lang = lang.lower()
        self.refresh_ui()

    def refresh_ui(self):
        # Update Sidebar
        self.btn_home.configure(text=self.tr("nav_home"))
        self.btn_modules.configure(text=self.tr("nav_store"))
        self.btn_monitor.configure(text=self.tr("nav_monitor"))
        self.btn_settings.configure(text=self.tr("nav_settings"))
        self.sidebar_separator.configure(text=self.tr("sidebar_installed"))
        self.status_label.configure(text=self.tr("status_ready"))

        # Re-load installed modules (buttons update)
        self.refresh_installed_modules()

        # Re-build ALL views (clearing cache mostly or just rebuilding current)
        self.views = {} 
        
        # Re-build static views
        self.build_home_view()
        self.build_modules_view()
        self.build_monitor_view()
        self.build_settings_view()

        # Restore current view
        if self.current_view_name:
            self.show_view(self.current_view_name)
        else:
            self.show_home_view()

    # --- Managers ---
    def load_installed(self):
        if os.path.exists(INSTALLED_MODULES_FILE):
            try:
                with open(INSTALLED_MODULES_FILE, 'r') as f:
                    return json.load(f)
            except:
                pass
        return []

    def save_installed(self):
        with open(INSTALLED_MODULES_FILE, 'w') as f:
            json.dump(self.installed_modules, f)

    def refresh_installed_modules(self):
        # Clear sidebar module buttons
        for widget in self.module_buttons_frame.winfo_children():
            widget.destroy()

        for mod_key in self.installed_modules:
            if mod_key in self.available_modules:
                # Instantiate if not already (keep instance to preserve state if possible)
                if mod_key not in self.loaded_modules:
                    cls = self.available_modules[mod_key]["class"]
                    self.loaded_modules[mod_key] = cls(self)
                
                # Add sidebar button
                mod_instance = self.loaded_modules[mod_key]
                # Mod title from metadata key
                title = self.tr(self.available_modules[mod_key]["title_key"])
                self.create_module_button(title, mod_key)

    def create_module_button(self, title, mod_key):
        btn = ctk.CTkButton(self.module_buttons_frame, text=title, 
                            command=lambda: self.open_module(mod_key), 
                            fg_color="transparent", text_color="gray90", 
                            hover_color=("gray70", "gray30"), anchor="w",
                            font=ctk.CTkFont(size=14, weight="bold"))
        btn.pack(fill="x", padx=20, pady=2)

    def open_module(self, mod_key):
        if mod_key in self.loaded_modules:
            # Recreate instance for language update (brute force but effective for text)
            cls = self.available_modules[mod_key]["class"]
            self.loaded_modules[mod_key] = cls(self) # Re-init module
            
            view = self.loaded_modules[mod_key].get_view()
            self.views[f"mod_{mod_key}"] = view
            self.show_view(f"mod_{mod_key}")

    def install_module(self, mod_key):
        if mod_key not in self.installed_modules:
            self.installed_modules.append(mod_key)
            self.save_installed()
            self.refresh_installed_modules()
            messagebox.showinfo(self.tr("btn_install"), self.tr("msg_installed"))
            self.refresh_ui()

    def create_nav_button(self, text_key, command, row):
        btn = ctk.CTkButton(self.sidebar_frame, text=self.tr(text_key), command=command, 
                            fg_color="transparent", text_color=("gray10", "#DCE4EE"), 
                            hover_color=("gray70", "gray30"), anchor="w",
                            font=ctk.CTkFont(size=16, weight="bold"))
        btn.grid(row=row, column=0, sticky="ew", padx=20, pady=10)
        return btn

    def show_view(self, view_name):
        if self.current_view:
            self.current_view.pack_forget()
        
        view = self.views.get(view_name)
        if view:
            view.pack(fill="both", expand=True)
            self.current_view = view
            self.current_view_name = view_name

    def show_home_view(self): self.show_view("home")
    def show_modules_view(self): self.show_view("modules")
    def show_monitor_view(self): self.show_view("monitor")
    def show_settings_view(self): self.show_view("settings")

    # --- Home View ---
    def build_home_view(self):
        # Create fresh frame
        frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.views["home"] = frame
        
        # Hero Section
        hero = ctk.CTkFrame(frame, fg_color=("gray90", "#111111"), corner_radius=15, border_width=2, border_color="#3B24C8")
        hero.pack(fill="x", pady=20, ipady=40)
        
        badge = ctk.CTkLabel(hero, text=self.tr("hero_badge"), fg_color="#3B24C8", text_color="white", corner_radius=20, font=ctk.CTkFont(weight="bold"))
        badge.pack(pady=(20, 10))
        
        title = ctk.CTkLabel(hero, text=self.tr("hero_title"), font=ctk.CTkFont(size=40, weight="bold"))
        title.pack(pady=5)
        
        subtitle = ctk.CTkLabel(hero, text=self.tr("hero_subtitle"), font=ctk.CTkFont(size=16), text_color="gray")
        subtitle.pack(pady=(0, 20))
        
        btn_box = ctk.CTkFrame(hero, fg_color="transparent")
        btn_box.pack(pady=10)
        
        action_btn = ctk.CTkButton(btn_box, text=self.tr("btn_explore"), command=self.show_modules_view, font=ctk.CTkFont(size=16, weight="bold"), height=40, width=180)
        action_btn.pack(side="left", padx=10)
        
        doc_btn = ctk.CTkButton(btn_box, text=self.tr("btn_docs"), fg_color="transparent", border_width=2, border_color="#3B24C8", text_color="#3B24C8", font=ctk.CTkFont(size=16, weight="bold"), height=40, width=180)
        doc_btn.pack(side="left", padx=10)

        # Features Grid
        grid = ctk.CTkFrame(frame, fg_color="transparent")
        grid.pack(fill="both", expand=True, pady=10)
        grid.grid_columnconfigure(0, weight=1)
        grid.grid_columnconfigure(1, weight=1)
        grid.grid_columnconfigure(2, weight=1)

        self.create_feature_card(grid, 0, "🚀", self.tr("feat_perf_title"), self.tr("feat_perf_desc"))
        self.create_feature_card(grid, 1, "🛡️", self.tr("feat_sec_title"), self.tr("feat_sec_desc"))
        self.create_feature_card(grid, 2, "🧩", self.tr("feat_mod_title"), self.tr("feat_mod_desc"))

    def create_feature_card(self, parent, col, icon, title, desc):
        card = ctk.CTkFrame(parent, fg_color=("white", "#1A1A1A"))
        card.grid(row=0, column=col, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(card, text=icon, font=ctk.CTkFont(size=30)).pack(pady=(20, 10))
        ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=18, weight="bold")).pack(pady=5)
        ctk.CTkLabel(card, text=desc, text_color="gray", wraplength=200).pack(pady=(0, 20), padx=10)

    # --- Modules View ---
    def build_modules_view(self):
        self.views["modules"] = ctk.CTkFrame(self.main_container, fg_color="transparent")
        frame = self.views["modules"]
        
        title = ctk.CTkLabel(frame, text=self.tr("store_title"), font=ctk.CTkFont(size=28, weight="bold"))
        title.pack(anchor="w", pady=(0, 20))
        
        scroll = ctk.CTkScrollableFrame(frame, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        for mod_key, meta in self.available_modules.items():
            card = ctk.CTkFrame(scroll, fg_color=("white", "#1A1A1A"), height=100)
            card.pack(fill="x", pady=10)
            
            icon_frame = ctk.CTkFrame(card, width=80, height=80, fg_color="#3B24C8")
            icon_frame.pack(side="left", padx=10, pady=10)
            icon_frame.pack_propagate(False) 
            ctk.CTkLabel(icon_frame, text=meta["icon"], text_color="white", font=ctk.CTkFont(size=24, weight="bold")).place(relx=0.5, rely=0.5, anchor="center")
            
            info_frame = ctk.CTkFrame(card, fg_color="transparent")
            info_frame.pack(side="left", padx=10, pady=10, fill="y")
            
            ctk.CTkLabel(info_frame, text=self.tr(meta["title_key"]), font=ctk.CTkFont(size=20, weight="bold")).pack(anchor="w")
            ctk.CTkLabel(info_frame, text=self.tr(meta["desc_key"]), text_color="gray").pack(anchor="w")
            
            btn_frame = ctk.CTkFrame(card, fg_color="transparent")
            btn_frame.pack(side="right", padx=20)
            
            if mod_key in self.installed_modules:
                ctk.CTkButton(btn_frame, text=self.tr("btn_open"), command=lambda k=mod_key: self.open_module(k), fg_color="green").pack()
            else:
                ctk.CTkButton(btn_frame, text=self.tr("btn_install"), command=lambda k=mod_key: self.install_module(k)).pack()

    # --- Monitor View ---
    def build_monitor_view(self):
        frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.views["monitor"] = frame
        
        ctk.CTkLabel(frame, text=self.tr("monitor_title"), font=ctk.CTkFont(size=28, weight="bold")).pack(anchor="w", pady=(0, 20))
        
        # Server Controls
        controls = ctk.CTkFrame(frame)
        controls.pack(fill="x", pady=10)
        
        ctk.CTkLabel(controls, text=self.tr("lbl_server_status")).pack(side="left", padx=20, pady=20)
        self.server_status_label = ctk.CTkLabel(controls, text=self.tr("status_stopped"), text_color="red", font=ctk.CTkFont(weight="bold"))
        self.server_status_label.pack(side="left", padx=10)
        
        self.btn_server_toggle = ctk.CTkButton(controls, text=self.tr("btn_start_server"), fg_color="green", command=self.toggle_server)
        self.btn_server_toggle.pack(side="right", padx=20)

        # Logs
        log_frame = ctk.CTkFrame(frame)
        log_frame.pack(fill="both", expand=True, pady=10)
        
        ctk.CTkLabel(log_frame, text=self.tr("lbl_logs"), font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=5)
        self.log_textbox = ctk.CTkTextbox(log_frame, font=ctk.CTkFont(family="Consolas"))
        self.log_textbox.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.log_textbox.insert("end", "> System initialized.\n")

    def toggle_server(self):
        current_text = self.server_status_label.cget("text")
        # Logic is simpler here just for UI toggling demo
        stopped_text = self.tr("status_stopped")
        running_text = self.tr("status_running")
        
        if current_text == stopped_text:
            self.server_status_label.configure(text=running_text, text_color="green")
            self.btn_server_toggle.configure(text=self.tr("btn_stop_server"), fg_color="red")
            self.log_textbox.insert("end", "> Server started.\n")
        else:
            self.server_status_label.configure(text=stopped_text, text_color="red")
            self.btn_server_toggle.configure(text=self.tr("btn_start_server"), fg_color="green")
            self.log_textbox.insert("end", "> Server stopped.\n")

    # --- Settings View ---
    def build_settings_view(self):
        frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.views["settings"] = frame
        
        ctk.CTkLabel(frame, text=self.tr("settings_title"), font=ctk.CTkFont(size=28, weight="bold")).pack(anchor="w", pady=(0, 20))
        ctk.CTkLabel(frame, text=self.tr("settings_desc"), text_color="gray").pack(anchor="w")

    def run(self):
        self.mainloop()

# Launcher function compatible with gateway.py
def launch_gui(profile_manager, preflight_report, log_dir, logo_path=None, manager=None, warm_callback=None, splash_path=None):
    app = StudioGUI(profile_manager, preflight_report, log_dir, logo_path, manager, warm_callback, splash_path)
    app.run()

def start_gui_thread(profile_manager, preflight_report, log_dir, logo_path=None, manager=None, warm_callback=None, splash_path=None):
    import threading
    thread = threading.Thread(
        target=launch_gui,
        args=(profile_manager, preflight_report, log_dir, logo_path, manager, warm_callback, splash_path),
        daemon=False 
    )
    thread.start()
    return thread

if __name__ == "__main__":
    # Mock data for standalone testing
    class MockProfileManager:
        def __init__(self):
            self.presets = []
            self.get_active_profile = lambda: ('default', {})
            self.system_info = {}
            
    mock_pm = MockProfileManager()
    mock_report = {"status": "Standalone Mode"}
    mock_log_dir = "logs"
    
    if not os.path.exists(mock_log_dir):
        os.makedirs(mock_log_dir)

    app = StudioGUI(mock_pm, mock_report, mock_log_dir)
    app.mainloop()
