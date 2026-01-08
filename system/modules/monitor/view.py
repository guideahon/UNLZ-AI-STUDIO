import customtkinter as ctk
import logging
import threading
import os
from pathlib import Path
from modules.base import StudioModule
from tkinter import messagebox
from PIL import Image

class MonitorModule(StudioModule):
    def __init__(self, parent):
        super().__init__(parent, "monitor", "System Monitor")
        self.view = None
        self.app = parent

    def get_view(self) -> ctk.CTkFrame:
        self.view = MonitorView(self.app.main_container, self.app)
        return self.view

    def on_enter(self):
        if self.view:
            self.view.update_status()

    def on_leave(self):
        pass

class MonitorView(ctk.CTkFrame):
    def __init__(self, master, app, **kwargs):
        super().__init__(master, **kwargs)
        self.app = app
        self.manager = app.manager
        self.pm = app.profile_manager
        self.services_ui = {} # Map service_key -> widgets dict
        self.model_map = {}
        self._refresh_job = None
        self._refresh_interval_ms = 3000
        
        try:
            tr = self.app.tr
            
            # --- Header ---
            header = ctk.CTkFrame(self, fg_color="transparent")
            header.pack(fill="x", pady=(0, 20))
            ctk.CTkLabel(header, text=tr("monitor_title"), font=ctk.CTkFont(size=28, weight="bold")).pack(side="left")
            
            # --- Hardware Info Grid ---
            hw_frame = ctk.CTkFrame(self)
            hw_frame.pack(fill="x", pady=10)
            
            sys_info = self.pm.system_info
            
            self.create_stat_card(hw_frame, "CPU", f"{sys_info.cpu_name}\n({sys_info.cpu_threads} Threads)", 0)
            self.create_stat_card(hw_frame, "RAM", f"{sys_info.ram_gb:.1f} GB", 1)
            
            gpu_text = "N/A"
            if sys_info.cuda_available:
                gpu_text = f"{sys_info.gpu_names[0]} ({sys_info.vram_gb_per_gpu[0]:.1f} GB)"
            self.create_stat_card(hw_frame, "GPU", gpu_text, 2)

            # --- Service Manager ---
            ctk.CTkLabel(self, text=tr("lbl_api_services"), font=ctk.CTkFont(size=20, weight="bold")).pack(anchor="w", pady=(20, 10))
            
            self.services_scroll = ctk.CTkScrollableFrame(self)
            self.services_scroll.pack(fill="both", expand=True)
            
            self.services = [
                {"key": "llm_service", "name": tr("svc_name_llm"), "desc": tr("svc_desc_llm"), "port": "8080"},
                {"key": "clm_service", "name": tr("svc_name_clm"), "desc": tr("svc_desc_clm"), "port": "8081"},
                {"key": "vlm_service", "name": tr("svc_name_vlm"), "desc": tr("svc_desc_vlm"), "port": "9090"},
                {"key": "alm_service", "name": tr("svc_name_alm"), "desc": tr("svc_desc_alm"), "port": "5000"},
                {"key": "slm_service", "name": tr("svc_name_slm"), "desc": tr("svc_desc_slm"), "port": "5001"},
            ]
            
            for srv in self.services:
                self.create_service_row(srv)

            self.update_status()
            self.schedule_refresh()

        except Exception as e:
            ctk.CTkLabel(self, text=self.app.tr("msg_monitor_error").format(e), text_color="red").pack()
            logging.error(f"MonitorView init error: {e}")

        self.bind("<Destroy>", self._on_destroy)

    def _on_destroy(self, event):
        if event.widget is self and self._refresh_job:
            try:
                self.after_cancel(self._refresh_job)
            except Exception:
                pass
            self._refresh_job = None

    def schedule_refresh(self):
        if not self.winfo_exists():
            return
        self._refresh_job = self.after(self._refresh_interval_ms, self._refresh_loop)

    def _refresh_loop(self):
        if not self.winfo_exists():
            return
        self.update_status()
        self.schedule_refresh()
    def create_stat_card(self, parent, title, value, col):
        card = ctk.CTkFrame(parent, fg_color=("gray90", "#202020"))
        card.grid(row=0, column=col, sticky="ew", padx=10, pady=10)
        parent.grid_columnconfigure(col, weight=1)
        
        ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=12, weight="bold"), text_color="gray").pack(pady=(10, 0))
        ctk.CTkLabel(card, text=value, font=ctk.CTkFont(size=16)).pack(pady=(5, 10))

    def create_service_row(self, srv):
        key = srv["key"]
        
        row = ctk.CTkFrame(self.services_scroll, fg_color=("gray95", "#2A2A2A"))
        row.pack(fill="x", pady=5, padx=5)
        
        # Info
        info = ctk.CTkFrame(row, fg_color="transparent")
        info.pack(side="left", padx=10, pady=10)
        ctk.CTkLabel(info, text=srv["name"], font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(info, text=srv["desc"], text_color="gray").pack(anchor="w")
        ctk.CTkLabel(info, text=f"Port: {srv['port']}", text_color="gray", font=ctk.CTkFont(size=10)).pack(anchor="w")
        
        # Controls
        ctrl = ctk.CTkFrame(row, fg_color="transparent")
        ctrl.pack(side="right", padx=10)
        
        # Model Selector for LLM Service
        file_combo = None
        if key == "llm_service" or key == "clm_service":
            file_combo = ctk.CTkComboBox(ctrl, width=200)
            file_combo.pack(side="left", padx=5)
            # Populate with models
            self.refresh_models(file_combo)
        
        # Status Label
        lbl_status = ctk.CTkLabel(ctrl, text=self.app.tr("status_unknown"), font=ctk.CTkFont(weight="bold"))
        lbl_status.pack(side="left", padx=10)
        
        # Buttons
        btn_install = ctk.CTkButton(ctrl, text=self.app.tr("btn_install"), width=80, command=lambda k=key: self.do_install(k))
        btn_uninstall = ctk.CTkButton(ctrl, text=self.app.tr("svc_uninstall"), width=80, fg_color="#991b1b", hover_color="#7f1d1d", command=lambda k=key: self.do_uninstall(k))
        btn_action = ctk.CTkButton(ctrl, text=self.app.tr("svc_start"), width=80, command=lambda k=key: self.toggle_service(k))
        
        # Store widgets to update later
        self.services_ui[key] = {
            "status": lbl_status,
            "btn_install": btn_install,
            "btn_uninstall": btn_uninstall,
            "btn_action": btn_action,
            "combo": file_combo
        }

    def refresh_models(self, combo):
        search_dir = Path(self.app.get_setting("model_dir", os.environ.get("LLAMA_MODEL_DIR", r"C:\models")))
        files = list(search_dir.rglob("*.gguf"))
        
        display_names = []
        self.model_map = {}
        recommended_suffix = self.app.tr("suffix_recommended")
        for f in files:
            name = f.name
            # Mark recommended models
            if "qwen2.5-coder-7b-instruct" in name.lower():
                name += recommended_suffix
            display_names.append(name)
            self.model_map[name] = f
            
        if not display_names: display_names = [self.app.tr("msg_no_models")]
        combo.configure(values=display_names)
        combo.set(display_names[0])

    def update_status(self):
        if not self.manager: return
        tr = self.app.tr

        for srv in self.services:
            key = srv["key"]
            ui = self.services_ui[key]
            
            try:
                state = self.manager.get_service_status(key)
                installed = state["installed"]
                running = state["running"]
                
                ui["btn_install"].pack_forget()
                ui["btn_uninstall"].pack_forget()
                ui["btn_action"].pack_forget()
                
                # Update generic buttons text
                ui["btn_install"].configure(text=tr("btn_install"))
                ui["btn_uninstall"].configure(text=tr("svc_uninstall"))

                if not installed:
                     ui["status"].configure(text=tr("svc_not_installed"), text_color="orange")
                     ui["btn_install"].pack(side="left", padx=5)
                
                if installed:
                    ui["btn_uninstall"].pack(side="right", padx=5)
                    
                    if running:
                        ui["status"].configure(text="● " + tr("svc_running"), text_color="green")
                        ui["btn_action"].configure(text=tr("svc_stop"), fg_color="red", state="normal", command=lambda k=key: self.toggle_service(k))
                        ui["btn_action"].pack(side="left", padx=5)
                        if ui["combo"]: ui["combo"].configure(state="disabled")
                    else:
                        ui["status"].configure(text="● " + tr("svc_stopped"), text_color="gray")
                        ui["btn_action"].configure(text=tr("svc_start"), fg_color="green", state="normal", command=lambda k=key: self.toggle_service(k))
                        ui["btn_action"].pack(side="left", padx=5)
                        if ui["combo"]: ui["combo"].configure(state="normal")

            except Exception as e:
                logging.error(f"Status update error for {key}: {e}")

    def do_install(self, key):
        # Update UI in main thread before starting background task
        # Keep simple "Disabled" state
        self.services_ui[key]["btn_install"].configure(state="disabled")
        threading.Thread(target=self._install_worker, args=(key,)).start()

    def _install_worker(self, key):
        tr = self.app.tr
        try:
            def progress(msg):
                logging.info(f"[{key}] {msg}")
            
            # Pass full key to manager
            self.manager.install_service(key, progress_callback=progress)
            self.after(0, lambda: messagebox.showinfo(tr("btn_install"), tr("msg_install_success").format(key)))
            
        except Exception as e:
            self.after(0, lambda: messagebox.showerror(tr("status_error"), tr("msg_install_failed").format(str(e))))
        finally:
             if self.services_ui[key]["combo"]:
                 self.after(0, lambda: self.refresh_models(self.services_ui[key]["combo"]))
             self.after(0, self.update_status)

    def do_uninstall(self, key):
        tr = self.app.tr
        if messagebox.askyesno(tr("svc_uninstall"), tr("msg_uninstall_confirm").format(key)):
            try:
                # Pass full key
                self.manager.uninstall_service(key)
                messagebox.showinfo(tr("svc_uninstall"), tr("msg_files_deleted"))
                self.update_status()
                if self.services_ui[key]["combo"]:
                    self.refresh_models(self.services_ui[key]["combo"])
            except Exception as e:
                messagebox.showerror(tr("status_error"), tr("msg_uninstall_failed").format(str(e)))

    def toggle_service(self, key):
        tr = self.app.tr
        if self.manager.is_running(key):
            # Visually set state to stopping/loading for feedback
            self.services_ui[key]["status"].configure(text="● " + tr("svc_loading"), text_color="orange")
            self.services_ui[key]["btn_action"].configure(state="disabled")
            
            self.manager.stop(key)
            self.after(500, self.update_status)
        else:
             config = {}
             # Default ports map
             ports = {
                 "llm_service": 8080, 
                 "clm_service": 8081,
                 "vlm_service": 9090, 
                 "alm_service": 5000,
                 "slm_service": 5001
             }
             config["port"] = ports.get(key, 8080)
             
             if key == "llm_service" or key == "clm_service":
                 model_label = self.services_ui[key]["combo"].get()
                 if model_label == tr("msg_no_models"):
                     messagebox.showerror("Error", tr("msg_model_not_found"))
                     return
                 model_path = self.model_map.get(model_label)
                 if not model_path:
                     # Fallback: strip suffix and rescan
                     clean_name = model_label.replace(tr("suffix_recommended"), "")
                     search_dir = Path(self.app.get_setting("model_dir", os.environ.get("LLAMA_MODEL_DIR", r"C:\models")))
                     for f in search_dir.rglob("*.gguf"):
                         if f.name == clean_name:
                             model_path = f
                             break
                 if not model_path:
                     messagebox.showerror("Error", tr("msg_model_not_found"))
                     return
                 config["model_path"] = str(model_path)
             
             # SET LOADING STATE VISUALLY
             self.services_ui[key]["status"].configure(text="● " + tr("svc_loading"), text_color="orange")
             self.services_ui[key]["btn_action"].configure(state="disabled")

             threading.Thread(target=self._start_bg, args=(key, config)).start()

    def _start_bg(self, key, config):
        try:
            self.manager.start_process(key, config)
            self.after(0, self.update_status)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror(self.app.tr("status_error"), self.app.tr("msg_start_failed").format(key, str(e))))
            self.after(0, self.update_status)

    def send_feedback(self):
        pass # Removed
