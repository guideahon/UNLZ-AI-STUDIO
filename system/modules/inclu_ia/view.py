import customtkinter as ctk
import os
import subprocess
import threading
import sys
import socket
import webbrowser
from modules.base import StudioModule

class IncluIAModule(StudioModule):
    def __init__(self, parent):
        super().__init__(parent, "inclu_ia", "Inclu-IA")
        self.view = None
        self.server_process = None
        self.app = parent

    def get_view(self) -> ctk.CTkFrame:
        self.view = IncluIAView(self.app.main_container, self, self.app)
        return self.view

    def on_enter(self):
        pass

    def on_leave(self):
        pass
        
    def start_server(self, model="tiny", port=5000):
        if self.server_process:
            return self.app.tr("msg_server_running")
            
        script_path = os.path.join(os.path.dirname(__file__), "software", "server.py")
        cmd = [sys.executable, script_path, "--model", model, "--port", str(port)]
        
        try:
            self.server_process = subprocess.Popen(
                cmd, 
                cwd=os.path.dirname(script_path),
                creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
            )
            return self.app.tr("msg_server_started").format(self.get_local_ip(), port)
        except Exception as e:
            return self.app.tr("msg_server_start_error").format(e)

    def stop_server(self):
        if self.server_process:
            self.server_process.terminate()
            self.server_process = None
            return self.app.tr("msg_server_stopped")
        return self.app.tr("msg_server_not_running")

    def get_local_ip(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"

class IncluIAView(ctk.CTkFrame):
    def __init__(self, master, module, app, **kwargs):
        super().__init__(master, **kwargs)
        self.module = module
        self.app = app
        tr = self.app.tr
        
        # Header
        ctk.CTkLabel(self, text=tr("mod_incluia_title"), font=ctk.CTkFont(size=24, weight="bold")).pack(pady=20)
        
        # Controls
        control_frame = ctk.CTkFrame(self)
        control_frame.pack(pady=10, padx=20, fill="x")
        
        ctk.CTkLabel(control_frame, text=tr("lbl_model_size")).pack(side="left", padx=10)
        self.model_var = ctk.StringVar(value="tiny")
        ctk.CTkOptionMenu(control_frame, variable=self.model_var, values=["tiny", "small", "base", "medium"]).pack(side="left", padx=10)
        
        self.btn_start = ctk.CTkButton(control_frame, text=tr("btn_start_server"), command=self.on_start, fg_color="green")
        self.btn_start.pack(side="right", padx=10, pady=10)
        
        self.btn_stop = ctk.CTkButton(control_frame, text=tr("btn_stop_server"), command=self.on_stop, fg_color="red", state="disabled")
        self.btn_stop.pack(side="right", padx=10, pady=10)
        
        # Info
        self.info_frame = ctk.CTkFrame(self, fg_color=("white", "#222"))
        self.info_frame.pack(pady=20, padx=20, fill="both", expand=True)
        
        self.status_label = ctk.CTkLabel(self.info_frame, text=f"Status: {tr('status_stopped')}", font=ctk.CTkFont(size=16))
        self.status_label.pack(pady=20)
        
        self.link_label = ctk.CTkLabel(self.info_frame, text="", font=ctk.CTkFont(size=20, weight="bold"), text_color="#EAB308")
        self.link_label.pack(pady=10)
        
        self.btn_open_browser = ctk.CTkButton(self.info_frame, text=tr("btn_open_browser"), command=self.open_browser, state="disabled")
        self.btn_open_browser.pack(pady=5)
        
        ctk.CTkLabel(self.info_frame, text=tr("lbl_server_link"), text_color="gray").pack(pady=5)

        # Check if already running
        if self.module.server_process:
            self.status_label.configure(text=f"Status: {tr('status_running')}", text_color="green")
            ip = self.module.get_local_ip()
            self.link_label.configure(text=f"http://{ip}:5000")
            self.btn_start.configure(state="disabled")
            self.btn_stop.configure(state="normal")
            self.btn_open_browser.configure(state="normal")

    def open_browser(self):
        ip = self.module.get_local_ip()
        url = f"http://{ip}:5000"
        webbrowser.open(url)

    def on_start(self):
        tr = self.app.tr
        msg = self.module.start_server(model=self.model_var.get())
        if self.module.server_process:
            self.status_label.configure(text=f"Status: {tr('status_running')}", text_color="green")
            ip = self.module.get_local_ip()
            self.link_label.configure(text=f"http://{ip}:5000")
            self.btn_start.configure(state="disabled")
            self.btn_stop.configure(state="normal")
            self.btn_open_browser.configure(state="normal")
        else:
            self.status_label.configure(text=f"Error: {msg}", text_color="red")

    def on_stop(self):
        tr = self.app.tr
        msg = self.module.stop_server()
        self.status_label.configure(text=f"Status: {tr('status_stopped')}", text_color="gray")
        self.link_label.configure(text="")
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.btn_open_browser.configure(state="disabled")
