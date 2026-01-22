import customtkinter as ctk
import os
import sys
import threading
import subprocess
import time
import logging
from pathlib import Path
from tkinter import filedialog, messagebox

from modules.base import StudioModule


class HYWorldModule(StudioModule):
    def __init__(self, parent):
        super().__init__(parent, "hyworld", "HunyuanWorld-Mirror")
        self.view = None
        self.app = parent

    def get_view(self) -> ctk.CTkFrame:
        self.view = HYWorldView(self.app.main_container, self.app)
        return self.view

    def on_enter(self):
        pass

    def on_leave(self):
        pass


class HYWorldView(ctk.CTkFrame):
    def __init__(self, master, app, **kwargs):
        super().__init__(master, **kwargs)
        self.app = app
        self.tr = app.tr
        self._busy = False
        self._current_action = None

        self.app_root = Path(__file__).resolve().parents[3]
        self.backend_dir = self.app_root / "system" / "ai-backends" / "HunyuanWorld-Mirror"
        self.output_dir = self.app_root / "system" / "hyworld-out"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._mode_options = {
            self.tr("hyworld_mode_demo"): "demo",
            self.tr("hyworld_mode_infer"): "infer",
        }
        self.mode_var = ctk.StringVar(value=self.tr("hyworld_mode_demo"))

        self.build_ui()
        self.refresh_buttons()

    def build_ui(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(10, 5))
        ctk.CTkLabel(header, text=self.tr("hyworld_title"), font=ctk.CTkFont(size=24, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(header, text=self.tr("hyworld_subtitle"), text_color="gray").pack(anchor="w")

        actions = ctk.CTkFrame(self, fg_color="transparent")
        actions.pack(fill="x", padx=10, pady=(10, 5))
        self.btn_install = ctk.CTkButton(actions, text=self.tr("hyworld_btn_install"), command=self.install_backend)
        self.btn_install.pack(side="left", padx=5)
        self.btn_uninstall = ctk.CTkButton(actions, text=self.tr("hyworld_btn_uninstall"), command=self.uninstall_backend)
        self.btn_uninstall.pack(side="left", padx=5)
        self.btn_deps = ctk.CTkButton(actions, text=self.tr("hyworld_btn_deps"), command=self.install_deps)
        self.btn_deps.pack(side="left", padx=5)
        self.btn_open = ctk.CTkButton(actions, text=self.tr("hyworld_btn_open_folder"), command=self.open_backend_folder)
        self.btn_open.pack(side="left", padx=5)

        links = ctk.CTkFrame(self, fg_color="transparent")
        links.pack(fill="x", padx=10, pady=(5, 5))
        ctk.CTkButton(links, text=self.tr("hyworld_btn_open_repo"), command=self.open_repo).pack(side="left", padx=5)
        ctk.CTkButton(links, text=self.tr("hyworld_btn_open_demo"), command=self.open_demo).pack(side="left", padx=5)

        status_frame = ctk.CTkFrame(self, fg_color="transparent")
        status_frame.pack(fill="x", padx=10, pady=(5, 0))
        ctk.CTkLabel(status_frame, text=self.tr("hyworld_status_label")).pack(side="left")
        self.status_value = ctk.CTkLabel(status_frame, text=self.tr("hyworld_status_idle"), text_color="gray")
        self.status_value.pack(side="left", padx=(6, 0))

        body = ctk.CTkFrame(self)
        body.pack(fill="both", expand=True, padx=10, pady=(10, 10))
        body.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(body, text=self.tr("hyworld_mode_label")).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 4))
        self.mode_menu = ctk.CTkOptionMenu(body, variable=self.mode_var, values=list(self._mode_options.keys()))
        self.mode_menu.grid(row=0, column=1, sticky="ew", padx=10, pady=(10, 4))

        ctk.CTkLabel(body, text=self.tr("hyworld_input_label")).grid(row=1, column=0, sticky="w", padx=10, pady=(0, 4))
        self.input_entry = ctk.CTkEntry(body, placeholder_text=self.tr("hyworld_input_placeholder"))
        self.input_entry.grid(row=1, column=1, sticky="ew", padx=10, pady=(0, 4))
        ctk.CTkButton(body, text=self.tr("hyworld_btn_browse"), command=self.select_input).grid(row=1, column=2, padx=(0, 10), pady=(0, 4))

        ctk.CTkLabel(body, text=self.tr("hyworld_output_label")).grid(row=2, column=0, sticky="w", padx=10, pady=(0, 4))
        self.output_entry = ctk.CTkEntry(body, placeholder_text=str(self.output_dir))
        self.output_entry.grid(row=2, column=1, sticky="ew", padx=10, pady=(0, 4))

        buttons = ctk.CTkFrame(body, fg_color="transparent")
        buttons.grid(row=3, column=0, columnspan=2, sticky="w", padx=10, pady=(10, 10))
        self.btn_download = ctk.CTkButton(buttons, text=self.tr("hyworld_btn_download_weights"), command=self.download_weights)
        self.btn_download.pack(side="left", padx=(0, 8))
        self.btn_run = ctk.CTkButton(buttons, text=self.tr("hyworld_btn_run"), command=self.run_action)
        self.btn_run.pack(side="left", padx=(0, 8))
        self.btn_open_output = ctk.CTkButton(buttons, text=self.tr("hyworld_btn_open_output"), command=self.open_output_folder)
        self.btn_open_output.pack(side="left")

        note = ctk.CTkLabel(self, text=self.tr("hyworld_note"), text_color="gray", wraplength=700, justify="left")
        note.pack(fill="x", padx=15, pady=(0, 10))

    def refresh_buttons(self):
        if self._busy:
            for btn in (
                self.btn_install, self.btn_uninstall, self.btn_deps, self.btn_open,
                self.btn_download, self.btn_run, self.btn_open_output
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
            if not self.btn_open.winfo_manager():
                self.btn_open.pack(side="left", padx=5)
        else:
            self.btn_uninstall.pack_forget()
            self.btn_deps.pack_forget()
            self.btn_open.pack_forget()
            if not self.btn_install.winfo_manager():
                self.btn_install.pack(side="left", padx=5)
        for btn in (
            self.btn_install, self.btn_uninstall, self.btn_deps, self.btn_open,
            self.btn_download, self.btn_run, self.btn_open_output
        ):
            btn.configure(state="normal")

    def set_busy(self, busy):
        self._busy = busy
        if self.status_value:
            self.status_value.configure(
                text=self.tr("status_in_progress") if busy else self.tr("hyworld_status_idle")
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
            self.log(self.tr("hyworld_msg_already_installed"))
            self.refresh_buttons()
            return
        if not self.check_git_available():
            messagebox.showwarning(self.tr("status_error"), self.tr("hyworld_msg_git_missing"))
            return
        self.backend_dir.parent.mkdir(parents=True, exist_ok=True)
        self.log(self.tr("hyworld_msg_installing"))
        self.set_busy(True)
        self._current_action = "install"
        self.run_process(
            ["git", "clone", "--depth", "1", "https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror", str(self.backend_dir)],
            on_done=self.on_process_done,
        )

    def uninstall_backend(self):
        if not self.backend_dir.exists():
            self.log(self.tr("hyworld_msg_not_installed"))
            self.refresh_buttons()
            return
        try:
            import shutil
            shutil.rmtree(self.backend_dir)
            self.log(self.tr("hyworld_msg_uninstalled"))
        except Exception as exc:
            self.log(f"{self.tr('status_error')}: {exc}")
        self.refresh_buttons()

    def install_deps(self):
        if not self.backend_dir.exists():
            messagebox.showwarning(self.tr("status_error"), self.tr("hyworld_msg_not_installed"))
            return
        mode_key = self._mode_options.get(self.mode_var.get(), "demo")
        req_name = "requirements_demo.txt" if mode_key == "demo" else "requirements.txt"
        req_path = self.backend_dir / req_name
        if not req_path.exists():
            messagebox.showwarning(self.tr("status_error"), self.tr("hyworld_msg_requirements_missing").format(req_name))
            return
        python_path = sys.executable.replace("pythonw.exe", "python.exe")
        self.log(self.tr("hyworld_msg_installing_deps").format(req_name))
        self.set_busy(True)
        self._current_action = "deps"
        self.run_process([python_path, "-m", "pip", "install", "-r", str(req_path)], on_done=self.on_process_done)

    def download_weights(self):
        if not self.backend_dir.exists():
            messagebox.showwarning(self.tr("status_error"), self.tr("hyworld_msg_not_installed"))
            return
        python_path = sys.executable.replace("pythonw.exe", "python.exe")
        self.log(self.tr("hyworld_msg_downloading_weights"))
        self.set_busy(True)
        self._current_action = "weights"
        self.run_process(
            [
                python_path,
                "-m",
                "huggingface_hub",
                "download",
                "tencent/HunyuanWorld-Mirror",
                "--local-dir",
                str(self.backend_dir / "ckpts"),
            ],
            on_done=self.on_process_done,
        )

    def run_action(self):
        if not self.backend_dir.exists():
            messagebox.showwarning(self.tr("status_error"), self.tr("hyworld_msg_not_installed"))
            return
        mode_key = self._mode_options.get(self.mode_var.get(), "demo")
        if mode_key == "demo":
            self.run_demo()
        else:
            self.run_inference()

    def run_demo(self):
        python_path = sys.executable.replace("pythonw.exe", "python.exe")
        app_path = self.backend_dir / "app.py"
        if not app_path.exists():
            messagebox.showwarning(self.tr("status_error"), self.tr("hyworld_msg_app_missing"))
            return
        self.log(self.tr("hyworld_msg_running_demo"))
        self.set_busy(True)
        self._current_action = "run"
        self.run_process([python_path, str(app_path)], on_done=self.on_process_done)

    def run_inference(self):
        input_path = self.input_entry.get().strip()
        if not input_path:
            messagebox.showwarning(self.tr("status_error"), self.tr("hyworld_msg_missing_input"))
            return
        output_dir = Path(self.output_entry.get().strip() or str(self.output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        script_path = self.backend_dir / "infer.py"
        if not script_path.exists():
            messagebox.showwarning(self.tr("status_error"), self.tr("hyworld_msg_infer_missing"))
            return
        python_path = sys.executable.replace("pythonw.exe", "python.exe")
        self.log(self.tr("hyworld_msg_running_infer"))
        self.set_busy(True)
        self._current_action = "run"
        self.run_process(
            [python_path, str(script_path), "--input_path", input_path, "--output_path", str(output_dir)],
            on_done=self.on_process_done,
        )

    def open_backend_folder(self):
        if self.backend_dir.exists():
            os.startfile(str(self.backend_dir))

    def open_output_folder(self):
        output_dir = Path(self.output_entry.get().strip() or str(self.output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        os.startfile(str(output_dir))

    def open_repo(self):
        import webbrowser
        webbrowser.open("https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror")

    def open_demo(self):
        import webbrowser
        webbrowser.open("https://huggingface.co/spaces/tencent/HunyuanWorld-Mirror")

    def select_input(self):
        mode_key = self._mode_options.get(self.mode_var.get(), "demo")
        if mode_key == "demo":
            messagebox.showinfo(self.tr("hyworld_title"), self.tr("hyworld_msg_demo_input_note"))
        file_path = filedialog.askopenfilename(
            title=self.tr("hyworld_dialog_input"),
            filetypes=[("Images/Videos", "*.png;*.jpg;*.jpeg;*.bmp;*.webp;*.mp4;*.mov;*.avi"), ("All files", "*.*")],
        )
        if file_path:
            self.input_entry.delete(0, "end")
            self.input_entry.insert(0, file_path)

    def run_process(self, cmd, on_done=None):
        def worker():
            try:
                process = subprocess.Popen(
                    cmd,
                    cwd=str(self.backend_dir) if self.backend_dir.exists() else None,
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
        self.set_busy(False)
        if returncode == 0:
            self.log(self.tr("hyworld_msg_done"))
        else:
            self.log(self.tr("hyworld_msg_failed").format(returncode))
        self._current_action = None
        self.refresh_buttons()
