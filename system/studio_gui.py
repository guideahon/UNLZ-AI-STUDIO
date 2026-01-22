import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import os
import sys
import json
import logging
import threading
import time
from pathlib import Path
from PIL import Image, ImageTk
import ctypes
import webbrowser
import psutil
import subprocess

# Ensure we can find modules in the current directory
sys.path.append(os.path.dirname(__file__))

# Import external Process Manager
try:
    from process_manager import GpuProcessManager
except ImportError:
    GpuProcessManager = None

print("DEBUG: LOADED LATEST VERSION OF STUDIO_GUI")

# Set AppUserModelID for Taskbar Icon
try:
    myappid = 'unlz.ai.studio.gui.1.0'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except Exception as e:
    print(f"Warning: Could not set AppUserModelID: {e}")

# Import modules
from modules.llm_frontend import LLMFrontendModule
from modules.inclu_ia import IncluIAModule
from modules.monitor import MonitorModule
from modules.research_assistant import ResearchAssistantModule
from modules.model_3d import Model3DModule
from modules.spotedit import SpotEditModule
from modules.finetune_glm import FinetuneGLMModule
from modules.hy_motion import HYMotionModule
from modules.proedit import ProEditModule
from modules.neutts import NeuttsModule
from modules.klein import KleinModule
from modules.hyworld import HYWorldModule
from modules.cyberscraper import CyberScraperModule
from modules.ml_sharp import MLSharpModule

# Set Theme
THEME_PATH = os.path.join(os.path.dirname(__file__), "assets", "themes", "ingenieria.json")
if os.path.exists(THEME_PATH):
    ctk.set_default_color_theme(THEME_PATH)
ctk.set_appearance_mode("Dark")

INSTALLED_MODULES_FILE = "installed_modules.json"
FAVORITES_FILE = "favorites_modules.json"
LANGUAGES_FILE = os.path.join(os.path.dirname(__file__), "assets", "languages.json")
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "data", "app_settings.json")

# Custom Logging Handler
class GuiLogHandler(logging.Handler):
    def __init__(self, app):
        super().__init__()
        self.app = app
    
    def emit(self, record):
        try:
            msg = self.format(record)
            self.app.log_queue.append(msg)
            # Push to global viewer if it exists
            if hasattr(self.app, 'log_viewer_widget') and self.app.log_viewer_widget:
                self.app.after(0, lambda: self.app.log_viewer_widget.append_log(msg))
        except Exception:
            self.handleError(record)

class LogViewerWidget(ctk.CTkFrame):
    def __init__(self, master, height=150, **kwargs):
        super().__init__(master, height=height, **kwargs)
        self.pack_propagate(False) # Strict height
        self.app = master
        
        # Header (Toggle)
        self.header = ctk.CTkFrame(self, height=30, fg_color=("gray85", "#202020"))
        self.header.pack(fill="x")
        self.header.pack_propagate(False)
        
        self.lbl_title = ctk.CTkLabel(self.header, text=self.app.tr("lbl_logs"), font=ctk.CTkFont(size=12, weight="bold"))
        self.lbl_title.pack(side="left", padx=10)
        
        self.btn_clear = ctk.CTkButton(self.header, text=self.app.tr("btn_clear"), width=60, height=20, command=self.clear_logs, font=ctk.CTkFont(size=10))
        self.btn_clear.pack(side="right", padx=5)
        
        # Textbox
        self.textbox = ctk.CTkTextbox(self, font=ctk.CTkFont(family="Consolas", size=10))
        self.textbox.pack(fill="both", expand=True, padx=2, pady=2)
        self.textbox.configure(state="disabled")

    def append_log(self, msg):
        self.textbox.configure(state="normal")
        self.textbox.insert("end", msg + "\n")
        self.textbox.see("end")
        self.textbox.configure(state="disabled")

    def clear_logs(self):
        self.textbox.configure(state="normal")
        self.textbox.delete("0.0", "end")
        self.textbox.configure(state="disabled")

class StudioGUI(ctk.CTk):
    def __init__(self, profile_manager, preflight_report, log_dir, logo_path=None, manager=None, warm_callback=None, splash_path=None):
        super().__init__()

        # Data & Managers
        self.profile_manager = profile_manager
        self.manager = manager
        if self.manager is None and GpuProcessManager:
            # Create a dedicated manager if one wasn't passed (e.g. standalone test)
            self.manager = GpuProcessManager(self.profile_manager, log_dir)
            
        self.log_dir = log_dir
        self.installed_modules = self.load_installed()
        self.favorite_modules = self.load_favorites()
        
        # Logging Setup
        self.log_queue = []
        self.log_viewer = None # Deprecated interface, keeping for compat
        self.log_viewer_widget = None # New widget
        
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        # Avoid duplicate handlers
        if not any(isinstance(h, GuiLogHandler) for h in self.logger.handlers):
            handler = GuiLogHandler(self)
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
        
        # i18n + Settings
        self.languages = self.load_languages()
        self.settings = self.load_settings()
        self.current_lang = (self.settings.get("language", "es") or "es").lower()
        if self.current_lang not in self.languages:
            self.current_lang = "es"

        saved_theme = self.settings.get("theme")
        if saved_theme in ("Dark", "Light"):
            ctk.set_appearance_mode(saved_theme)

        # Assets Paths
        self.assets_path = os.path.join(os.path.dirname(__file__), "assets")
        self.icon_path = os.path.join(self.assets_path, "SOLO-LOGO-AZUL-HORIZONTAL-fondo-transparente.ico")
        self.logo_img_path = os.path.join(self.assets_path, "LOGO AZUL HORIZONTAL - fondo transparente.png")
        self.status_logo_path = os.path.join(self.assets_path, "SOLO LOGO AZUL HORIZONTAL - fondo transparente.png")
        self.status_matrix_image = self._build_status_matrix_image()

        # Window Setup
        self.title("UNLZ AI Studio")
        self.geometry("1280x800")
        self.minsize(1024, 768)
        
        if os.path.exists(self.icon_path):
            self.iconbitmap(self.icon_path)
            
        # Grid Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        # Row 1 will be for the sticky footer
        self.grid_rowconfigure(1, weight=0) 

        # --- Sidebar ---
        self.sidebar_frame = ctk.CTkFrame(self, width=240, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=2, sticky="nsew") # Sidebar spans full height
        self.sidebar_frame.grid_rowconfigure(6, weight=1)

        # Logo Image
        if os.path.exists(self.logo_img_path):
            pil_img = Image.open(self.logo_img_path)
            self.logo_image = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(204, 100))
            self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="", image=self.logo_image)
            self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 30))
        else:
            self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="UNLZ AI STUDIO", font=ctk.CTkFont(size=22, weight="bold"))
            self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 30))

        # Nav Buttons
        self.btn_home = self.create_nav_button("nav_home", self.show_home_view, row=1)
        self.btn_modules = self.create_nav_button("nav_store", self.show_modules_view, row=2)
        self.btn_settings = self.create_nav_button("nav_settings", self.show_settings_view, row=4)
        
        self.sidebar_separator = ctk.CTkLabel(self.sidebar_frame, text=self.tr("sidebar_installed"), font=ctk.CTkFont(size=12, weight="bold"), text_color=("black", "gray"))
        self.sidebar_separator.grid(row=5, column=0, sticky="w", padx=20, pady=(20, 5))
        
        # Dynamic Module Buttons Container
        self.module_buttons_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.module_buttons_frame.grid(row=6, column=0, sticky="nsew")
        self.module_buttons_frame.grid_columnconfigure(0, weight=1)
        
        # Language Switcher
        self.footer_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.footer_frame.grid(row=7, column=0, padx=20, pady=20, sticky="ew")
        
        self.lang_switcher = ctk.CTkSegmentedButton(self.footer_frame, values=["ES", "EN"], command=self.change_language)
        self.lang_switcher.set("EN" if self.current_lang == "en" else "ES")
        self.lang_switcher.pack(fill="x", pady=(0, 10))
        
        # Theme Switcher
        self.theme_switcher = ctk.CTkSegmentedButton(self.footer_frame, values=[self.tr("theme_dark"), self.tr("theme_light")], command=self.change_theme)
        self.theme_switcher.set(self.tr("theme_dark") if ctk.get_appearance_mode() == "Dark" else self.tr("theme_light"))
        self.theme_switcher.pack(fill="x", pady=(0, 10))

        self.status_label = ctk.CTkLabel(self.footer_frame, text=self.tr("status_ready"), font=ctk.CTkFont(size=12), text_color="gray")
        self.status_label.pack()

        # --- Main Content Area ---
        self.main_container = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_container.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        
        # --- Global Log Footer ---
        self.log_viewer_widget = LogViewerWidget(self, height=180, fg_color=("gray95", "#111"))
        self.log_viewer_widget.grid(row=1, column=1, sticky="ew", padx=20, pady=(0, 20))
        if not self.settings.get("show_logs", False):
            self.log_viewer_widget.grid_remove()
        
        # Toggle Log Button in Sidebar
        self.btn_toggle_logs = ctk.CTkButton(self.footer_frame, text=self.tr("btn_toggle_logs"), command=self.toggle_logs, height=24, fg_color="gray", font=ctk.CTkFont(size=11))
        self.btn_toggle_logs.pack(pady=5)

        # Views
        self.views = {}
        self.current_view_name = None
        self.current_view = None

        # Available Modules
        self.available_modules = {
            "monitor": {
                "class": MonitorModule,
                "title_key": "mod_monitor_title", 
                "desc_key": "mod_monitor_desc",
                "icon": "Sys"
            },
            "ml_sharp": {
                "class": MLSharpModule,
                "title_key": "mod_mlsharp_title",
                "desc_key": "mod_mlsharp_desc",
                "icon": "MS"
            },
            "model_3d": {
                "class": Model3DModule,
                "title_key": "mod_model3d_title",
                "desc_key": "mod_model3d_desc",
                "icon": "3D"
            },
            "llm_frontend": {
                "class": LLMFrontendModule,
                "title_key": "mod_llm_title",
                "desc_key": "mod_llm_desc",
                "icon": "AI"
            },
            "finetune_glm": {
                "class": FinetuneGLMModule,
                "title_key": "mod_finetune_glm_title",
                "desc_key": "mod_finetune_glm_desc",
                "icon": "FT"
            },
            "inclu_ia": {
                "class": IncluIAModule,
                "title_key": "mod_incluia_title",
                "desc_key": "mod_incluia_desc",
                "icon": "CC"
            },
            "research_assistant": {
                "class": ResearchAssistantModule,
                "title_key": "mod_research_title",
                "desc_key": "mod_research_desc",
                "icon": "RA"
            },
            "spotedit": {
                "class": SpotEditModule,
                "title_key": "mod_spotedit_title",
                "desc_key": "mod_spotedit_desc",
                "icon": "SE"
            },
            "klein": {
                "class": KleinModule,
                "title_key": "mod_klein_title",
                "desc_key": "mod_klein_desc",
                "icon": "K2"
            },
            "hyworld": {
                "class": HYWorldModule,
                "title_key": "mod_hyworld_title",
                "desc_key": "mod_hyworld_desc",
                "icon": "HW"
            },
            "cyberscraper": {
                "class": CyberScraperModule,
                "title_key": "mod_cyber_title",
                "desc_key": "mod_cyber_desc",
                "icon": "CS"
            },
            "hy_motion": {
                "class": HYMotionModule,
                "title_key": "mod_hymotion_title",
                "desc_key": "mod_hymotion_desc",
                "icon": "HM"
            },
            "neutts": {
                "class": NeuttsModule,
                "title_key": "mod_neutts_title",
                "desc_key": "mod_neutts_desc",
                "icon": "TTS"
            }
        }
        
        # Loaded module instances
        self.loaded_modules = {} 
        self._home_telemetry_job = None
        self._home_telemetry_frame = None
        self._home_gpu_util = None
        self._home_gpu_util_counter = 0
        
        # Default Install Monitor if not present
        if "monitor" not in self.installed_modules:
            self.installed_modules.insert(0, "monitor") # Prepend
        
        # Initial Build
        self.refresh_ui()
        self.show_home_view()
        
        logging.info("Application Started.")

    def toggle_logs(self):
        if self.log_viewer_widget.winfo_viewable():
            self.log_viewer_widget.grid_remove()
            self.set_setting("show_logs", False)
        else:
            self.log_viewer_widget.grid()
            self.set_setting("show_logs", True)

    def _build_status_matrix_image(self):
        if not os.path.exists(self.status_logo_path):
            return None
        try:
            tile = Image.open(self.status_logo_path).convert("RGBA")
            tile.thumbnail((8, 8))
            tile_w, tile_h = tile.size
            cols = 3
            rows = 3
            padding = 2
            grid_w = cols * tile_w + (cols - 1) * padding
            grid_h = rows * tile_h + (rows - 1) * padding
            canvas = Image.new("RGBA", (grid_w, grid_h), (0, 0, 0, 0))
            for row in range(rows):
                for col in range(cols):
                    x = col * (tile_w + padding)
                    y = row * (tile_h + padding)
                    canvas.paste(tile, (x, y), tile)
            return ctk.CTkImage(light_image=canvas, dark_image=canvas, size=(grid_w, grid_h))
        except Exception:
            return None

    def get_status_matrix_image(self):
        return self.status_matrix_image

    def register_log_viewer(self, viewer):
        pass # Deprecated

    # --- i18n Helpers ---
    def load_languages(self):
        if os.path.exists(LANGUAGES_FILE):
            try:
                with open(LANGUAGES_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {}

    def load_settings(self):
        settings_dir = os.path.dirname(SETTINGS_FILE)
        if not os.path.exists(settings_dir):
            os.makedirs(settings_dir)
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
            except Exception:
                pass
        return {}

    def save_settings(self):
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=2)
        except Exception:
            pass

    def get_setting(self, key, default=None):
        return self.settings.get(key, default)

    def set_setting(self, key, value):
        self.settings[key] = value
        self.save_settings()

    def tr(self, key):
        if self.current_lang in self.languages:
            return self.languages[self.current_lang].get(key, key)
        return key

    def change_language(self, lang):
        self.current_lang = lang.lower()
        self.set_setting("language", self.current_lang)
        self.refresh_ui()

    def change_theme(self, mode):
        theme_key = mode
        if mode == self.tr("theme_dark"):
            theme_key = "Dark"
        elif mode == self.tr("theme_light"):
            theme_key = "Light"
        ctk.set_appearance_mode(theme_key)
        self.set_setting("theme", theme_key)

    def refresh_ui(self):
        if self._home_telemetry_job:
            try:
                self.after_cancel(self._home_telemetry_job)
            except Exception:
                pass
            self._home_telemetry_job = None
            self._home_telemetry_frame = None

        self.btn_home.configure(text=self.tr("nav_home"))
        self.btn_modules.configure(text=self.tr("nav_store"))
        self.btn_settings.configure(text=self.tr("nav_settings"))
        self.sidebar_separator.configure(text=self.tr("sidebar_installed"))
        self.status_label.configure(text=self.tr("status_ready"))

        self.refresh_installed_modules()

        # Save current view info before clearing
        target_view = self.current_view_name

        self.views = {} 
        self.build_home_view()
        self.build_modules_view()
        self.build_settings_view()

        # Restore View
        if target_view and target_view.startswith("mod_"):
            mod_key = target_view.replace("mod_", "")
            if mod_key in self.loaded_modules:
                self.open_module(mod_key)
                return

        if target_view and target_view in self.views:
            self.show_view(target_view)
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

    def load_favorites(self):
        if os.path.exists(FAVORITES_FILE):
            try:
                with open(FAVORITES_FILE, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return [str(x) for x in data][:3]
            except:
                pass
        return []

    def save_installed(self):
        with open(INSTALLED_MODULES_FILE, 'w') as f:
            json.dump(self.installed_modules, f)

    def save_favorites(self):
        with open(FAVORITES_FILE, 'w') as f:
            json.dump(self.favorite_modules[:3], f)

    def refresh_installed_modules(self):
        for widget in self.module_buttons_frame.winfo_children():
            widget.destroy()

        ordered = []
        for key in self.favorite_modules:
            if key in self.installed_modules:
                ordered.append(key)
        for key in self.installed_modules:
            if key not in ordered:
                ordered.append(key)

        for mod_key in ordered:
            if mod_key in self.available_modules:
                if mod_key not in self.loaded_modules:
                    cls = self.available_modules[mod_key]["class"]
                    self.loaded_modules[mod_key] = cls(self)
                
                title = self.tr(self.available_modules[mod_key]["title_key"])
                self.create_module_button(title, mod_key)

    def create_module_button(self, title, mod_key):
        if mod_key in self.favorite_modules:
            title = f"* {title}"
        btn = ctk.CTkButton(self.module_buttons_frame, text=title, 
                            command=lambda: self.open_module(mod_key), 
                            fg_color="transparent", text_color=("gray10", "gray90"), 
                            hover_color=("gray70", "gray30"), anchor="w",
                            font=ctk.CTkFont(size=14, weight="bold"))
        btn.pack(fill="x", padx=20, pady=2)
        btn.bind("<Button-3>", lambda event, key=mod_key: self.show_favorite_menu(event, key))

    def show_favorite_menu(self, event, mod_key):
        menu = tk.Menu(self, tearoff=0)
        if mod_key in self.favorite_modules:
            menu.add_command(label=self.tr("fav_remove"), command=lambda: self.toggle_favorite(mod_key, False))
        else:
            menu.add_command(label=self.tr("fav_add"), command=lambda: self.toggle_favorite(mod_key, True))
        menu.tk_popup(event.x_root, event.y_root)

    def toggle_favorite(self, mod_key, add):
        if add:
            if mod_key in self.favorite_modules:
                return
            if len(self.favorite_modules) >= 3:
                messagebox.showwarning(self.tr("status_error"), self.tr("fav_limit"))
                return
            self.favorite_modules.append(mod_key)
        else:
            if mod_key in self.favorite_modules:
                self.favorite_modules.remove(mod_key)
        self.save_favorites()
        self.refresh_installed_modules()

    def open_module(self, mod_key):
        logging.info(f"Request to open module: {mod_key}")
        
        # Check if installed
        if mod_key not in self.installed_modules:
             if mod_key in self.available_modules:
                 if messagebox.askyesno("Module Required", f"The module '{self.tr(self.available_modules[mod_key]['title_key'])}' is not installed.\nDo you want to install it now?"):
                     self.install_module(mod_key)
                 return
             else:
                 messagebox.showerror("Error", f"Module {mod_key} not found.")
                 return

        # Lazy load if needed
        if mod_key not in self.loaded_modules and mod_key in self.available_modules:
            logging.info(f"Lazy loading module: {mod_key}")
            try:
                cls = self.available_modules[mod_key]["class"]
                self.loaded_modules[mod_key] = cls(self)
            except Exception as e:
                logging.error(f"Failed to load module {mod_key}: {e}")
                messagebox.showerror("Error", f"Failed to load module code: {e}")
                return
                
        if mod_key in self.loaded_modules:
            try:
                # Use existing instance to preserve state
                module = self.loaded_modules[mod_key]
                
                # Get fresh view (updates i18n)
                view = module.get_view()
                self.views[f"mod_{mod_key}"] = view
                self.show_view(f"mod_{mod_key}")
                logging.info(f"Opened view for {mod_key}")
            except Exception as e:
                 logging.error(f"Error opening module {mod_key}: {e}")
                 messagebox.showerror("Error", f"Failed to open module view: {e}")
        else:
             messagebox.showerror("Error", f"Module {mod_key} could not be loaded.")

    def install_module(self, mod_key):
        if mod_key not in self.installed_modules:
            self.installed_modules.append(mod_key)
            self.save_installed()
            self.refresh_installed_modules()
            messagebox.showinfo(self.tr("btn_install"), f"{mod_key}: " + self.tr("msg_installed"))
            self.refresh_ui()
            if messagebox.askyesno("Open", "Desea abrir el módulo ahora?"):
                self.open_module(mod_key)

    def uninstall_module(self, mod_key):
        if mod_key in self.installed_modules:
            if messagebox.askyesno("Uninstall", f"Seguro que desea desinstalar {mod_key}?"):
                self.installed_modules.remove(mod_key)
                self.save_installed()
                if mod_key in self.loaded_modules:
                    del self.loaded_modules[mod_key]
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
        
        doc_btn = ctk.CTkButton(btn_box, text=self.tr("btn_docs"), command=self.open_docs, fg_color="transparent", border_width=2, border_color="#3B24C8", text_color="#3B24C8", font=ctk.CTkFont(size=16, weight="bold"), height=40, width=180)
        doc_btn.pack(side="left", padx=10)

        # Telemetry
        telemetry = ctk.CTkFrame(frame, fg_color=("white", "#1A1A1A"), corner_radius=12)
        telemetry.pack(fill="x", pady=(10, 20))
        telemetry.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(telemetry, text=self.tr("home_telemetry_title"), font=ctk.CTkFont(size=18, weight="bold")).grid(row=0, column=0, columnspan=2, sticky="w", padx=15, pady=(10, 5))

        self.home_cpu_label = ctk.CTkLabel(telemetry, text="", font=ctk.CTkFont(size=14))
        self.home_cpu_label.grid(row=1, column=0, sticky="w", padx=15, pady=5)

        self.home_ram_label = ctk.CTkLabel(telemetry, text="", font=ctk.CTkFont(size=14))
        self.home_ram_label.grid(row=1, column=1, sticky="w", padx=15, pady=5)

        gpu_text = self.tr("home_telemetry_na")
        sys_info = self.profile_manager.system_info
        if sys_info.cuda_available:
            gpu_text = f"{sys_info.gpu_names[0]} ({sys_info.vram_gb_per_gpu[0]:.1f} GB)"
        self.home_gpu_label = ctk.CTkLabel(telemetry, text=f"{self.tr('home_telemetry_gpu')}: {gpu_text}", font=ctk.CTkFont(size=14))
        self.home_gpu_label.grid(row=2, column=0, columnspan=2, sticky="w", padx=15, pady=(0, 10))

        self._home_telemetry_frame = frame
        self.update_home_telemetry()

        # Features Grid
        grid = ctk.CTkFrame(frame, fg_color="transparent")
        grid.pack(fill="both", expand=True, pady=10)
        grid.grid_columnconfigure(0, weight=1)
        grid.grid_columnconfigure(1, weight=1)
        grid.grid_columnconfigure(2, weight=1)

        self.create_feature_card(grid, 0, "🚀", self.tr("feat_perf_title"), self.tr("feat_perf_desc"))
        self.create_feature_card(grid, 1, "🔒", self.tr("feat_sec_title"), self.tr("feat_sec_desc"))
        self.create_feature_card(grid, 2, "🧩", self.tr("feat_mod_title"), self.tr("feat_mod_desc"))

    def create_feature_card(self, parent, col, icon, title, desc):
        card = ctk.CTkFrame(parent, fg_color=("white", "#1A1A1A"))
        card.grid(row=0, column=col, padx=10, pady=10, sticky="nsew")
        
        # Center content frame explicitly using place
        content_frame = ctk.CTkFrame(card, fg_color="transparent")
        content_frame.place(relx=0.5, rely=0.5, anchor="center")

        # Pack items inside the centered frame
        ctk.CTkLabel(content_frame, text=icon, font=ctk.CTkFont(size=30), anchor="center").pack(pady=(20, 10))
        ctk.CTkLabel(content_frame, text=title, font=ctk.CTkFont(size=18, weight="bold"), anchor="center", justify="center").pack(pady=5)
        # Using justify=center for alignment
        ctk.CTkLabel(content_frame, text=desc, text_color="gray", wraplength=200, justify="center", anchor="center").pack(pady=(0, 20), padx=10)

    def update_home_telemetry(self):
        if not self._home_telemetry_frame or not self._home_telemetry_frame.winfo_exists():
            return
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        ram_text = f"{mem.used / (1024**3):.1f} / {mem.total / (1024**3):.1f} GB"

        cpu_name = self.profile_manager.system_info.cpu_name
        if cpu_name:
            cpu_text = f"{cpu_name} - {cpu:.0f}%"
        else:
            cpu_text = f"{cpu:.0f}%"
        self.home_cpu_label.configure(text=f"{self.tr('home_telemetry_cpu')}: {cpu_text}")
        self.home_ram_label.configure(text=f"{self.tr('home_telemetry_ram')}: {ram_text}")

        self._home_gpu_util_counter = (self._home_gpu_util_counter + 1) % 3
        if self._home_gpu_util_counter == 0:
            self._home_gpu_util = self.get_gpu_utilization()

        sys_info = self.profile_manager.system_info
        gpu_text = self.tr("home_telemetry_na")
        if sys_info.cuda_available:
            base = f"{sys_info.gpu_names[0]} ({sys_info.vram_gb_per_gpu[0]:.1f} GB)"
            if self._home_gpu_util is not None:
                gpu_text = f"{base} - {self._home_gpu_util}%"
            else:
                gpu_text = base
        self.home_gpu_label.configure(text=f"{self.tr('home_telemetry_gpu')}: {gpu_text}")

        self._home_telemetry_job = self.after(1000, self.update_home_telemetry)

    def get_gpu_utilization(self):
        try:
            creationflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                creationflags=creationflags
            )
            if proc.returncode != 0:
                return None
            line = proc.stdout.strip().splitlines()
            if not line:
                return None
            return int(line[0].strip())
        except Exception:
            return None

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
            
            # Icon
            icon_frame = ctk.CTkFrame(card, width=80, height=80, fg_color="#3B24C8")
            icon_frame.pack(side="left", padx=10, pady=10)
            icon_frame.pack_propagate(False) 
            ctk.CTkLabel(icon_frame, text=meta["icon"], text_color="white", font=ctk.CTkFont(size=24, weight="bold")).place(relx=0.5, rely=0.5, anchor="center")
            
            # Info
            info_frame = ctk.CTkFrame(card, fg_color="transparent")
            info_frame.pack(side="left", padx=10, pady=10, fill="y")
            
            ctk.CTkLabel(info_frame, text=self.tr(meta["title_key"]), font=ctk.CTkFont(size=20, weight="bold")).pack(anchor="w")
            ctk.CTkLabel(info_frame, text=self.tr(meta["desc_key"]), text_color="gray").pack(anchor="w")
            
            # Buttons
            btn_frame = ctk.CTkFrame(card, fg_color="transparent")
            btn_frame.pack(side="right", padx=20)
            
            if mod_key in self.installed_modules:
                ctk.CTkButton(btn_frame, text=self.tr("btn_open"), command=lambda k=mod_key: self.open_module(k), fg_color="green", width=100).pack(pady=2)
                if mod_key != "monitor": # Prevent uninstalling Monitor
                    ctk.CTkButton(btn_frame, text="Uninstall", command=lambda k=mod_key: self.uninstall_module(k), fg_color="#991b1b", hover_color="#7f1d1d", width=100).pack(pady=2)
            else:
                ctk.CTkButton(btn_frame, text=self.tr("btn_install"), command=lambda k=mod_key: self.install_module(k), width=100).pack()

    # --- Settings View ---
    def build_settings_view(self):
        frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.views["settings"] = frame
        
        ctk.CTkLabel(frame, text=self.tr("settings_title"), font=ctk.CTkFont(size=28, weight="bold")).pack(anchor="w", pady=(0, 20))
        ctk.CTkLabel(frame, text=self.tr("settings_desc"), text_color="gray").pack(anchor="w")

    def open_docs(self):
        try:
            docs_path = Path(os.path.dirname(__file__)).parent / "readme.md"
            if docs_path.exists():
                webbrowser.open(docs_path.as_uri())
            else:
                messagebox.showwarning("Docs", self.tr("msg_docs_not_found"))
        except Exception as e:
            messagebox.showerror("Docs", f"{self.tr('msg_docs_open_error')}: {e}")

    def run(self):
        self.mainloop()

# Launcher
def launch_gui(profile_manager, preflight_report, log_dir, logo_path=None, manager=None, warm_callback=None, splash_path=None):
    app = StudioGUI(profile_manager, preflight_report, log_dir, logo_path, manager, warm_callback, splash_path)
    app.run()

if __name__ == "__main__":
    try:
        # LOGGING STARTUP ERRORS
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logging.basicConfig(filename=os.path.join(log_dir, "startup.log"), level=logging.DEBUG)
        
        logging.info("Starting up...")
        
        # --- SPLASH SCREEN ---
        splash_root = tk.Tk()
        splash_root.overrideredirect(True) # No title bar
        
        # Center splash
        w = 500
        h = 250
        ws = splash_root.winfo_screenwidth()
        hs = splash_root.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        splash_root.geometry('%dx%d+%d+%d' % (w, h, x, y))
        
        # Logo
        logo_path = os.path.join(os.path.dirname(__file__), "assets", "LOGO AZUL HORIZONTAL - fondo transparente.png")
        if os.path.exists(logo_path):
             # Resize for splash
             from PIL import Image, ImageTk
             pil_img = Image.open(logo_path)
             # maintain aspect ratio
             pil_img.thumbnail((400, 200))
             img = ImageTk.PhotoImage(pil_img)
             lbl = tk.Label(splash_root, image=img, bg="white")
             lbl.pack(expand=True, fill="both")
             splash_root.configure(bg="white")
        else:
             tk.Label(splash_root, text="UNLZ AI STUDIO\nLoading...", font=("Arial", 20)).pack(expand=True)
             
        splash_root.update()
        
        # --- LOADING CORE ---
        logging.info("Importing runtime_profiles...")
        from runtime_profiles import detect_system_info, ProfileManager
        
        logging.info("Detecting System Info...")
        sys_info = detect_system_info()
        data_dir = Path(os.path.join(os.path.dirname(__file__), "data"))
        
        logging.info("Initializing Profile Manager...")
        pm = ProfileManager(sys_info, data_dir)
        
        report = {"status": "Standalone Runtime", "details": sys_info.to_display_dict()}
        
        logging.info("Loading GUI...")
        time.sleep(1) # Simulated delay to see splash
        
        splash_root.destroy()
        
        app = StudioGUI(pm, report, log_dir)
        app.mainloop()
        
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        logging.critical(f"Startup Crash: {err_msg}")
        
        # Show error in UI since console might be hidden
        try:
            if 'splash_root' in locals():
                splash_root.destroy()
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Startup Error", f"Failed to start application:\n{e}\n\nCheck logs/startup.log for details.")
        except:
            pass
        print(err_msg)
