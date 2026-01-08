import customtkinter as ctk
import os
import sys
import threading
import subprocess
import webbrowser
from pathlib import Path
from tkinter import messagebox

from modules.base import StudioModule


class SpotEditModule(StudioModule):
    def __init__(self, parent):
        super().__init__(parent, "spotedit", "SpotEdit")
        self.view = None
        self.app = parent

    def get_view(self) -> ctk.CTkFrame:
        self.view = SpotEditView(self.app.main_container, self.app)
        return self.view

    def on_enter(self):
        pass

    def on_leave(self):
        pass


class SpotEditView(ctk.CTkFrame):
    def __init__(self, master, app, **kwargs):
        super().__init__(master, **kwargs)
        self.app = app
        self.tr = app.tr

        self.app_root = Path(__file__).resolve().parents[4]
        self.backend_dir = self.app_root / "system" / "ai-backends" / "SpotEdit"

        self.build_ui()
        self.refresh_buttons()

    def build_ui(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(10, 5))

        ctk.CTkLabel(header, text=self.tr("spotedit_title"), font=ctk.CTkFont(size=24, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(header, text=self.tr("spotedit_subtitle"), text_color="gray").pack(anchor="w")

        actions = ctk.CTkFrame(self, fg_color="transparent")
        actions.pack(fill="x", padx=10, pady=(10, 5))

        self.btn_install = ctk.CTkButton(actions, text=self.tr("spotedit_btn_install"), command=self.install_backend)
        self.btn_install.pack(side="left", padx=5)

        self.btn_uninstall = ctk.CTkButton(actions, text=self.tr("spotedit_btn_uninstall"), command=self.uninstall_backend)
        self.btn_uninstall.pack(side="left", padx=5)

        self.btn_deps = ctk.CTkButton(actions, text=self.tr("spotedit_btn_deps"), command=self.install_deps)
        self.btn_deps.pack(side="left", padx=5)

        self.btn_open = ctk.CTkButton(actions, text=self.tr("spotedit_btn_open_folder"), command=self.open_backend_folder)
        self.btn_open.pack(side="left", padx=5)

        links = ctk.CTkFrame(self, fg_color="transparent")
        links.pack(fill="x", padx=10, pady=(5, 5))

        ctk.CTkButton(links, text=self.tr("spotedit_btn_open_repo"), command=self.open_repo).pack(side="left", padx=5)
        ctk.CTkButton(links, text=self.tr("spotedit_btn_open_page"), command=self.open_page).pack(side="left", padx=5)

        log_frame = ctk.CTkFrame(self)
        log_frame.pack(fill="both", expand=True, padx=10, pady=(10, 10))
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(log_frame, text=self.tr("spotedit_log_label")).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 0))
        self.log_box = ctk.CTkTextbox(log_frame, font=ctk.CTkFont(family="Consolas", size=11))
        self.log_box.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.log_box.configure(state="disabled")

    def refresh_buttons(self):
        installed = self.backend_dir.exists()
        if installed:
            self.btn_install.pack_forget()
            if not self.btn_uninstall.winfo_manager():
                self.btn_uninstall.pack(side="left", padx=5)
            if not self.btn_deps.winfo_manager():
                self.btn_deps.pack(side="left", padx=5)
            if not self.btn_open.winfo_manager():
                self.btn_open.pack(side="left", padx=5)
        else:
            self.btn_uninstall.pack_forget()
            self.btn_deps.pack_forget()
            self.btn_open.pack_forget()
            if not self.btn_install.winfo_manager():
                self.btn_install.pack(side="left", padx=5)

    def install_backend(self):
        if self.backend_dir.exists():
            self.log(self.tr("spotedit_msg_already_installed"))
            self.refresh_buttons()
            return
        if not self.check_git_available():
            messagebox.showwarning(self.tr("status_error"), self.tr("spotedit_msg_git_missing"))
            return
        self.backend_dir.parent.mkdir(parents=True, exist_ok=True)
        self.log(self.tr("spotedit_msg_installing"))
        self.run_process(["git", "clone", "--depth", "1", "https://github.com/Biangbiang0321/SpotEdit", str(self.backend_dir)])
        self.refresh_buttons()

    def uninstall_backend(self):
        if not self.backend_dir.exists():
            self.log(self.tr("spotedit_msg_not_installed"))
            self.refresh_buttons()
            return
        try:
            import shutil
            shutil.rmtree(self.backend_dir)
            self.log(self.tr("spotedit_msg_uninstalled"))
        except Exception as exc:
            self.log(f"{self.tr('status_error')}: {exc}")
        self.refresh_buttons()

    def install_deps(self):
        if not self.backend_dir.exists():
            messagebox.showwarning(self.tr("status_error"), self.tr("spotedit_msg_not_installed"))
            return
        req_path = self.backend_dir / "requirements.txt"
        if not req_path.exists():
            messagebox.showwarning(self.tr("status_error"), self.tr("spotedit_msg_requirements_missing"))
            return
        python_path = sys.executable.replace("pythonw.exe", "python.exe")
        self.log(self.tr("spotedit_msg_installing_deps"))
        self.run_process([python_path, "-m", "pip", "install", "-r", str(req_path)])

    def open_backend_folder(self):
        if self.backend_dir.exists():
            os.startfile(str(self.backend_dir))

    def open_repo(self):
        webbrowser.open("https://github.com/Biangbiang0321/SpotEdit")

    def open_page(self):
        webbrowser.open("https://biangbiang0321.github.io/SpotEdit.github.io/")

    def check_git_available(self):
        try:
            result = subprocess.run(["git", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return result.returncode == 0
        except Exception:
            return False

    def run_process(self, cmd):
        def worker():
            try:
                process = subprocess.Popen(
                    cmd,
                    cwd=str(self.backend_dir.parent),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
                )
                for line in process.stdout:
                    self.safe_log(line.rstrip())
                process.wait()
                if process.returncode == 0:
                    self.safe_log(self.tr("spotedit_msg_done"))
                else:
                    self.safe_log(self.tr("spotedit_msg_failed").format(process.returncode))
            except Exception as exc:
                self.safe_log(f"{self.tr('status_error')}: {exc}")

        threading.Thread(target=worker, daemon=True).start()

    def log(self, message):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", message + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def safe_log(self, message):
        self.after(0, lambda: self.log(message))
