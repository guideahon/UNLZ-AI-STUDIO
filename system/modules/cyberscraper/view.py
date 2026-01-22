import customtkinter as ctk
import os
import sys
import threading
import subprocess
import logging
from pathlib import Path
from tkinter import messagebox

from modules.base import StudioModule


class CyberScraperModule(StudioModule):
    def __init__(self, parent):
        super().__init__(parent, "cyberscraper", "CyberScraper 2077")
        self.view = None
        self.app = parent

    def get_view(self) -> ctk.CTkFrame:
        self.view = CyberScraperView(self.app.main_container, self.app)
        return self.view

    def on_enter(self):
        pass

    def on_leave(self):
        pass


class CyberScraperView(ctk.CTkFrame):
    def __init__(self, master, app, **kwargs):
        super().__init__(master, **kwargs)
        self.app = app
        self.tr = app.tr
        self._busy = False
        self._current_action = None
        self._server_process = None

        self.app_root = Path(__file__).resolve().parents[3]
        self.backend_dir = self.app_root / "system" / "ai-backends" / "CyberScraper-2077"

        self._branch_options = {
            self.tr("cyber_branch_main"): "",
            self.tr("cyber_branch_scrapeless"): "CyberScrapeless-2077",
        }
        self.branch_var = ctk.StringVar(value=self.tr("cyber_branch_main"))

        self.build_ui()
        self.refresh_buttons()

    def build_ui(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(10, 5))
        ctk.CTkLabel(header, text=self.tr("cyber_title"), font=ctk.CTkFont(size=24, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(header, text=self.tr("cyber_subtitle"), text_color="gray").pack(anchor="w")

        actions = ctk.CTkFrame(self, fg_color="transparent")
        actions.pack(fill="x", padx=10, pady=(10, 5))
        self.btn_install = ctk.CTkButton(actions, text=self.tr("cyber_btn_install"), command=self.install_backend)
        self.btn_install.pack(side="left", padx=5)
        self.btn_uninstall = ctk.CTkButton(actions, text=self.tr("cyber_btn_uninstall"), command=self.uninstall_backend)
        self.btn_uninstall.pack(side="left", padx=5)
        self.btn_deps = ctk.CTkButton(actions, text=self.tr("cyber_btn_deps"), command=self.install_deps)
        self.btn_deps.pack(side="left", padx=5)
        self.btn_playwright = ctk.CTkButton(actions, text=self.tr("cyber_btn_playwright"), command=self.install_playwright)
        self.btn_playwright.pack(side="left", padx=5)
        self.btn_open = ctk.CTkButton(actions, text=self.tr("cyber_btn_open_folder"), command=self.open_backend_folder)
        self.btn_open.pack(side="left", padx=5)

        links = ctk.CTkFrame(self, fg_color="transparent")
        links.pack(fill="x", padx=10, pady=(5, 5))
        ctk.CTkButton(links, text=self.tr("cyber_btn_open_repo"), command=self.open_repo).pack(side="left", padx=5)
        ctk.CTkButton(links, text=self.tr("cyber_btn_open_ui"), command=self.open_ui).pack(side="left", padx=5)

        status_frame = ctk.CTkFrame(self, fg_color="transparent")
        status_frame.pack(fill="x", padx=10, pady=(5, 0))
        ctk.CTkLabel(status_frame, text=self.tr("cyber_status_label")).pack(side="left")
        self.status_value = ctk.CTkLabel(status_frame, text=self.tr("cyber_status_idle"), text_color="gray")
        self.status_value.pack(side="left", padx=(6, 0))

        form = ctk.CTkFrame(self)
        form.pack(fill="both", expand=True, padx=10, pady=(10, 10))
        form.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(form, text=self.tr("cyber_branch_label")).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 4))
        self.branch_menu = ctk.CTkOptionMenu(form, variable=self.branch_var, values=list(self._branch_options.keys()))
        self.branch_menu.grid(row=0, column=1, sticky="ew", padx=10, pady=(10, 4))

        ctk.CTkLabel(form, text=self.tr("cyber_port_label")).grid(row=1, column=0, sticky="w", padx=10, pady=(0, 4))
        self.port_entry = ctk.CTkEntry(form)
        self.port_entry.insert(0, "8501")
        self.port_entry.grid(row=1, column=1, sticky="ew", padx=10, pady=(0, 4))

        ctk.CTkLabel(form, text=self.tr("cyber_openai_label")).grid(row=2, column=0, sticky="w", padx=10, pady=(0, 4))
        self.openai_entry = ctk.CTkEntry(form, show="*")
        self.openai_entry.grid(row=2, column=1, sticky="ew", padx=10, pady=(0, 4))

        ctk.CTkLabel(form, text=self.tr("cyber_google_label")).grid(row=3, column=0, sticky="w", padx=10, pady=(0, 4))
        self.google_entry = ctk.CTkEntry(form, show="*")
        self.google_entry.grid(row=3, column=1, sticky="ew", padx=10, pady=(0, 4))

        ctk.CTkLabel(form, text=self.tr("cyber_scrapeless_label")).grid(row=4, column=0, sticky="w", padx=10, pady=(0, 4))
        self.scrapeless_entry = ctk.CTkEntry(form, show="*")
        self.scrapeless_entry.grid(row=4, column=1, sticky="ew", padx=10, pady=(0, 4))

        ctk.CTkLabel(form, text=self.tr("cyber_ollama_label")).grid(row=5, column=0, sticky="w", padx=10, pady=(0, 4))
        self.ollama_entry = ctk.CTkEntry(form)
        self.ollama_entry.grid(row=5, column=1, sticky="ew", padx=10, pady=(0, 4))

        buttons = ctk.CTkFrame(form, fg_color="transparent")
        buttons.grid(row=6, column=0, columnspan=2, sticky="w", padx=10, pady=(10, 10))
        self.btn_start = ctk.CTkButton(buttons, text=self.tr("cyber_btn_start"), command=self.start_server)
        self.btn_start.pack(side="left", padx=(0, 8))
        self.btn_stop = ctk.CTkButton(buttons, text=self.tr("cyber_btn_stop"), command=self.stop_server)
        self.btn_stop.pack(side="left", padx=(0, 8))

        note = ctk.CTkLabel(self, text=self.tr("cyber_note"), text_color="gray", wraplength=720, justify="left")
        note.pack(fill="x", padx=15, pady=(0, 10))

    def refresh_buttons(self):
        if self._busy:
            for btn in (
                self.btn_install, self.btn_uninstall, self.btn_deps, self.btn_playwright,
                self.btn_open, self.btn_start, self.btn_stop
            ):
                btn.configure(state="disabled")
            return

        installed = self.backend_dir.exists()
        if installed:
            self.btn_install.pack_forget()
            if not self.btn_uninstall.winfo_manager():
                self.btn_uninstall.pack(side="left", padx=5)
            if not self.btn_deps.winfo_manager():
                self.btn_deps.pack(side="left", padx=5)
            if not self.btn_playwright.winfo_manager():
                self.btn_playwright.pack(side="left", padx=5)
            if not self.btn_open.winfo_manager():
                self.btn_open.pack(side="left", padx=5)
        else:
            self.btn_uninstall.pack_forget()
            self.btn_deps.pack_forget()
            self.btn_playwright.pack_forget()
            self.btn_open.pack_forget()
            if not self.btn_install.winfo_manager():
                self.btn_install.pack(side="left", padx=5)

        server_running = self._server_process is not None
        self.btn_start.configure(state="disabled" if server_running else "normal")
        self.btn_stop.configure(state="normal" if server_running else "disabled")
        for btn in (self.btn_install, self.btn_uninstall, self.btn_deps, self.btn_playwright, self.btn_open):
            btn.configure(state="normal")

    def set_busy(self, busy, status_key=None):
        self._busy = busy
        if self.status_value:
            if status_key:
                self.status_value.configure(text=self.tr(status_key))
            else:
                self.status_value.configure(
                    text=self.tr("status_in_progress") if busy else self.tr("cyber_status_idle")
                )
        self.refresh_buttons()

    def log(self, message):
        logging.info(message)

    def safe_log(self, message):
        self.after(0, lambda: logging.info(message))

    def check_git_available(self):
        try:
            result = subprocess.run(["git", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return result.returncode == 0
        except Exception:
            return False

    def install_backend(self):
        if self.backend_dir.exists():
            self.log(self.tr("cyber_msg_already_installed"))
            self.refresh_buttons()
            return
        if not self.check_git_available():
            messagebox.showwarning(self.tr("status_error"), self.tr("cyber_msg_git_missing"))
            return
        branch = self._branch_options.get(self.branch_var.get(), "")
        cmd = ["git", "clone", "--depth", "1"]
        if branch:
            cmd.extend(["-b", branch])
        cmd.extend(["https://github.com/itsOwen/CyberScraper-2077.git", str(self.backend_dir)])
        self.backend_dir.parent.mkdir(parents=True, exist_ok=True)
        self.log(self.tr("cyber_msg_installing"))
        self.set_busy(True)
        self._current_action = "install"
        self.run_process(cmd, on_done=self.on_process_done)

    def uninstall_backend(self):
        if not self.backend_dir.exists():
            self.log(self.tr("cyber_msg_not_installed"))
            self.refresh_buttons()
            return
        try:
            import shutil
            shutil.rmtree(self.backend_dir)
            self.log(self.tr("cyber_msg_uninstalled"))
        except Exception as exc:
            self.log(f"{self.tr('status_error')}: {exc}")
        self.refresh_buttons()

    def install_deps(self):
        if not self.backend_dir.exists():
            messagebox.showwarning(self.tr("status_error"), self.tr("cyber_msg_not_installed"))
            return
        req_path = self.backend_dir / "requirements.txt"
        if not req_path.exists():
            messagebox.showwarning(self.tr("status_error"), self.tr("cyber_msg_requirements_missing"))
            return
        python_path = sys.executable.replace("pythonw.exe", "python.exe")
        self.log(self.tr("cyber_msg_installing_deps"))
        self.set_busy(True)
        self._current_action = "deps"
        self.run_process([python_path, "-m", "pip", "install", "-r", str(req_path)], on_done=self.on_process_done)

    def install_playwright(self):
        if not self.backend_dir.exists():
            messagebox.showwarning(self.tr("status_error"), self.tr("cyber_msg_not_installed"))
            return
        python_path = sys.executable.replace("pythonw.exe", "python.exe")
        self.log(self.tr("cyber_msg_installing_playwright"))
        self.set_busy(True)
        self._current_action = "playwright"
        self.run_process([python_path, "-m", "playwright", "install"], on_done=self.on_process_done)

    def start_server(self):
        if self._server_process:
            messagebox.showinfo(self.tr("cyber_title"), self.tr("cyber_msg_server_running"))
            return
        if not self.backend_dir.exists():
            messagebox.showwarning(self.tr("status_error"), self.tr("cyber_msg_not_installed"))
            return
        port = self.port_entry.get().strip() or "8501"
        if not port.isdigit():
            messagebox.showwarning(self.tr("status_error"), self.tr("cyber_msg_invalid_port"))
            return

        python_path = sys.executable.replace("pythonw.exe", "python.exe")
        cmd = [python_path, "-m", "streamlit", "run", "main.py", "--server.port", port]
        env = os.environ.copy()
        self.apply_env(env)

        self.log(self.tr("cyber_msg_starting"))
        self.set_busy(True, status_key="cyber_status_starting")
        self._current_action = "server"

        def worker():
            try:
                self._server_process = subprocess.Popen(
                    cmd,
                    cwd=str(self.backend_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
                )
                self.after(0, lambda: self.set_busy(False, status_key="cyber_status_running"))
                for line in self._server_process.stdout:
                    self.safe_log(line.rstrip())
                self._server_process.wait()
            except Exception as exc:
                self.safe_log(f"{self.tr('status_error')}: {exc}")
            finally:
                self.after(0, self.on_server_exit)

        threading.Thread(target=worker, daemon=True).start()

    def stop_server(self):
        if not self._server_process:
            return
        self.log(self.tr("cyber_msg_stopping"))
        try:
            self._server_process.terminate()
            self._server_process.wait(timeout=5)
        except Exception:
            try:
                self._server_process.kill()
            except Exception:
                pass
        self._server_process = None
        self.set_busy(False, status_key="cyber_status_idle")

    def on_server_exit(self):
        self._server_process = None
        self.set_busy(False, status_key="cyber_status_idle")
        self.log(self.tr("cyber_msg_server_stopped"))

    def apply_env(self, env):
        openai_key = self.openai_entry.get().strip()
        google_key = self.google_entry.get().strip()
        scrapeless_key = self.scrapeless_entry.get().strip()
        ollama_url = self.ollama_entry.get().strip()
        if openai_key:
            env["OPENAI_API_KEY"] = openai_key
        if google_key:
            env["GOOGLE_API_KEY"] = google_key
        if scrapeless_key:
            env["SCRAPELESS_API_KEY"] = scrapeless_key
        if ollama_url:
            env["OLLAMA_BASE_URL"] = ollama_url

    def open_backend_folder(self):
        if self.backend_dir.exists():
            os.startfile(str(self.backend_dir))

    def open_repo(self):
        import webbrowser
        webbrowser.open("https://github.com/itsOwen/CyberScraper-2077")

    def open_ui(self):
        port = self.port_entry.get().strip() or "8501"
        import webbrowser
        webbrowser.open(f"http://localhost:{port}")

    def run_process(self, cmd, on_done=None):
        def worker():
            try:
                process = subprocess.Popen(
                    cmd,
                    cwd=str(self.backend_dir.parent) if self._current_action == "install" else str(self.backend_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
                )
                for line in process.stdout:
                    self.safe_log(line.rstrip())
                process.wait()
                returncode = process.returncode
            except Exception as exc:
                returncode = 1
                self.safe_log(f"{self.tr('status_error')}: {exc}")
            if on_done:
                self.after(0, lambda: on_done(returncode))
            else:
                self.after(0, lambda: self.on_process_done(returncode))

        threading.Thread(target=worker, daemon=True).start()

    def on_process_done(self, returncode):
        self.set_busy(False, status_key="cyber_status_idle")
        if returncode == 0:
            self.log(self.tr("cyber_msg_done"))
        else:
            self.log(self.tr("cyber_msg_failed").format(returncode))
        self._current_action = None
        self.refresh_buttons()
