import customtkinter as ctk
import os
import sys
import threading
import subprocess
import webbrowser
import time
import logging
import importlib.metadata
from pathlib import Path
from tkinter import messagebox, filedialog

from modules.base import StudioModule


class HYMotionModule(StudioModule):
    def __init__(self, parent):
        super().__init__(parent, "hy_motion", "HY-Motion 1.0")
        self.view = None
        self.app = parent

    def get_view(self) -> ctk.CTkFrame:
        self.view = HYMotionView(self.app.main_container, self.app)
        return self.view

    def on_enter(self):
        pass

    def on_leave(self):
        pass


class HYMotionView(ctk.CTkFrame):
    def __init__(self, master, app, **kwargs):
        super().__init__(master, **kwargs)
        self.app = app
        self.tr = app.tr
        self._busy = False
        self._current_action = None

        # parents[3] points to the app root (UNLZ-AI-STUDIO)
        self.app_root = Path(__file__).resolve().parents[3]
        self.backend_dir = self.resolve_backend_dir()
        self.output_dir = self.app_root / "system" / "hymotion-out"
        self._deps_installed = False
        self._model_options = {
            self.tr("hymotion_model_full"): "HY-Motion-1.0",
            self.tr("hymotion_model_lite"): "HY-Motion-1.0-Lite",
        }
        self.model_var = ctk.StringVar(value=self.tr("hymotion_model_lite"))

        self.build_ui()
        self.refresh_buttons()

    def build_ui(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(10, 5))

        ctk.CTkLabel(header, text=self.tr("hymotion_title"), font=ctk.CTkFont(size=24, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(header, text=self.tr("hymotion_subtitle"), text_color="gray").pack(anchor="w")

        actions = ctk.CTkFrame(self, fg_color="transparent")
        actions.pack(fill="x", padx=10, pady=(10, 5))

        self.btn_install = ctk.CTkButton(actions, text=self.tr("hymotion_btn_install"), command=self.install_backend)
        self.btn_install.pack(side="left", padx=5)

        self.btn_uninstall = ctk.CTkButton(actions, text=self.tr("hymotion_btn_uninstall"), command=self.uninstall_backend)
        self.btn_uninstall.pack(side="left", padx=5)

        self.btn_deps = ctk.CTkButton(actions, text=self.tr("hymotion_btn_deps"), command=self.install_deps)
        self.btn_deps.pack(side="left", padx=5)

        self.btn_open = ctk.CTkButton(actions, text=self.tr("hymotion_btn_open_folder"), command=self.open_backend_folder)
        self.btn_open.pack(side="left", padx=5)

        links = ctk.CTkFrame(self, fg_color="transparent")
        links.pack(fill="x", padx=10, pady=(5, 5))

        ctk.CTkButton(links, text=self.tr("hymotion_btn_open_repo"), command=self.open_repo).pack(side="left", padx=5)

        status_frame = ctk.CTkFrame(self, fg_color="transparent")
        status_frame.pack(fill="x", padx=10, pady=(5, 0))
        ctk.CTkLabel(status_frame, text=self.tr("hymotion_status_label")).pack(side="left")
        self.status_value = ctk.CTkLabel(status_frame, text=self.tr("hymotion_status_idle"), text_color="gray")
        self.status_value.pack(side="left", padx=(6, 0))

        usage = ctk.CTkFrame(self)
        usage.pack(fill="both", expand=True, padx=10, pady=(10, 10))
        usage.grid_columnconfigure(0, weight=1)
        usage.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(usage, text=self.tr("hymotion_usage_title"), font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 0))

        ctk.CTkLabel(usage, text=self.tr("hymotion_model_label")).grid(row=1, column=0, sticky="w", padx=10, pady=(6, 0))
        self.model_menu = ctk.CTkOptionMenu(usage, variable=self.model_var, values=list(self._model_options.keys()))
        self.model_menu.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))

        ctk.CTkLabel(usage, text=self.tr("hymotion_prompt_label")).grid(row=3, column=0, sticky="w", padx=10, pady=(0, 0))
        self.prompt_text = ctk.CTkTextbox(
            usage,
            height=120,
            fg_color=("gray95", "#1f1f1f"),
            border_width=1,
            border_color=("gray70", "#2a2a2a"),
        )
        self.prompt_text.grid(row=4, column=0, sticky="nsew", padx=10, pady=(0, 10))

        ctk.CTkLabel(usage, text=self.tr("hymotion_output_label")).grid(row=5, column=0, sticky="w", padx=10, pady=(0, 0))
        self.output_entry = ctk.CTkEntry(usage, placeholder_text=str(self.output_dir))
        self.output_entry.grid(row=6, column=0, sticky="ew", padx=10, pady=(0, 10))

        buttons = ctk.CTkFrame(usage, fg_color="transparent")
        buttons.grid(row=7, column=0, sticky="w", padx=10, pady=(0, 10))
        ctk.CTkButton(buttons, text=self.tr("hymotion_btn_download_weights"), command=self.download_weights).pack(side="left", padx=(0, 8))
        ctk.CTkButton(buttons, text=self.tr("hymotion_btn_run"), command=self.run_generation).pack(side="left", padx=(0, 8))
        ctk.CTkButton(buttons, text=self.tr("hymotion_btn_open_output"), command=self.open_output_folder).pack(side="left")

    def refresh_buttons(self):
        self.backend_dir = self.resolve_backend_dir()
        if self._busy:
            for btn in (self.btn_install, self.btn_uninstall, self.btn_deps, self.btn_open):
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
            self._deps_installed = self.detect_deps_installed()
            self.btn_deps.configure(
                text=self.tr("hymotion_btn_deps_installed") if self._deps_installed else self.tr("hymotion_btn_deps")
            )
        else:
            self.btn_uninstall.pack_forget()
            self.btn_deps.pack_forget()
            self.btn_open.pack_forget()
            if not self.btn_install.winfo_manager():
                self.btn_install.pack(side="left", padx=5)
            self._deps_installed = False
        for btn in (self.btn_install, self.btn_uninstall, self.btn_deps, self.btn_open):
            btn.configure(state="normal")

    def set_busy(self, busy):
        self._busy = busy
        if self.status_value:
            self.status_value.configure(
                text=self.tr("status_in_progress") if busy else self.tr("hymotion_status_idle")
            )
        self.refresh_buttons()

    def install_backend(self):
        if self.backend_dir.exists():
            self.log(self.tr("hymotion_msg_already_installed"))
            self.refresh_buttons()
            return
        if not self.check_git_available():
            messagebox.showwarning(self.tr("status_error"), self.tr("hymotion_msg_git_missing"))
            return
        self.backend_dir.parent.mkdir(parents=True, exist_ok=True)
        self.log(self.tr("hymotion_msg_installing"))
        self.set_busy(True)
        self._current_action = "install"
        self.run_process(
            ["git", "clone", "--depth", "1", "https://github.com/Tencent-Hunyuan/HY-Motion-1.0", str(self.backend_dir)],
            on_done=self.on_process_done,
        )

    def uninstall_backend(self):
        if not self.backend_dir.exists():
            self.log(self.tr("hymotion_msg_not_installed"))
            self.refresh_buttons()
            return
        try:
            import shutil
            shutil.rmtree(self.backend_dir)
            self.log(self.tr("hymotion_msg_uninstalled"))
        except Exception as exc:
            self.log(f"{self.tr('status_error')}: {exc}")
        self.refresh_buttons()

    def install_deps(self):
        if not self.backend_dir.exists():
            messagebox.showwarning(self.tr("status_error"), self.tr("hymotion_msg_not_installed"))
            return
        req_path = self.backend_dir / "requirements.txt"
        if not req_path.exists():
            messagebox.showwarning(self.tr("status_error"), self.tr("hymotion_msg_requirements_missing"))
            return
        if self._deps_installed:
            if not messagebox.askyesno(self.tr("hymotion_reinstall_title"), self.tr("hymotion_reinstall_msg")):
                return
        python_path = sys.executable.replace("pythonw.exe", "python.exe")
        self.log(self.tr("hymotion_msg_installing_deps"))
        self.set_busy(True)
        self._current_action = "deps"
        self.run_process(
            [python_path, "-m", "pip", "install", "--only-binary=:all:", "PyYAML==6.0.2"],
            on_done=lambda code: self.on_deps_prereq_done(code, req_path),
        )

    def open_backend_folder(self):
        if self.backend_dir.exists():
            os.startfile(str(self.backend_dir))

    def open_repo(self):
        webbrowser.open("https://github.com/Tencent-Hunyuan/HY-Motion-1.0")

    def check_git_available(self):
        try:
            result = subprocess.run(["git", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return result.returncode == 0
        except Exception:
            return False

    def run_process(self, cmd, on_done=None):
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
                buffer = []
                while True:
                    ch = process.stdout.read(1)
                    if ch == "":
                        break
                    if ch in ("\n", "\r"):
                        if buffer:
                            self.safe_log("".join(buffer).rstrip())
                            buffer = []
                        continue
                    buffer.append(ch)
                process.wait()
                returncode = process.returncode
                if returncode == 0:
                    self.safe_log(self.tr("hymotion_msg_done"))
                else:
                    self.safe_log(self.tr("hymotion_msg_failed").format(process.returncode))
            except Exception as exc:
                returncode = 1
                self.safe_log(f"{self.tr('status_error')}: {exc}")
            if on_done:
                self.after(0, lambda: on_done(returncode))

        threading.Thread(target=worker, daemon=True).start()

    def log(self, message):
        logging.info(message)

    def safe_log(self, message):
        self.after(0, lambda: logging.info(message))

    def on_process_done(self, returncode):
        self.set_busy(False)
        if returncode == 0 and self._current_action == "deps":
            self.log(self.tr("hymotion_msg_deps_done"))
            try:
                (self.backend_dir / ".deps_installed").write_text("ok", encoding="utf-8")
            except Exception:
                pass
        self._current_action = None
        self.refresh_buttons()

    def on_deps_prereq_done(self, returncode, req_path):
        if returncode != 0:
            self.set_busy(False)
            self.log(self.tr("hymotion_msg_failed").format(returncode))
            return
        python_path = sys.executable.replace("pythonw.exe", "python.exe")
        self.run_process([python_path, "-m", "pip", "install", "-r", str(req_path)], on_done=self.on_process_done)

    def download_weights(self):
        if not self.backend_dir.exists():
            messagebox.showwarning(self.tr("status_error"), self.tr("hymotion_msg_not_installed"))
            return
        model_key = self._model_options.get(self.model_var.get(), "HY-Motion-1.0-Lite")
        script_path = self.app_root / "system" / "data" / "hymotion" / "hymotion_download.py"
        python_path = sys.executable.replace("pythonw.exe", "python.exe")
        self.log(self.tr("hymotion_msg_downloading_weights").format(model_key))
        self.set_busy(True)
        self._current_action = "weights"
        self.run_process(
            [
                python_path,
                "-u",
                str(script_path),
                "--model",
                model_key,
                "--output_dir",
                str(self.backend_dir / "ckpts" / "tencent"),
            ],
            on_done=self.on_weights_done,
        )

    def on_weights_done(self, returncode):
        self.set_busy(False)
        if returncode == 0:
            self.log(self.tr("hymotion_msg_weights_done"))
        else:
            self.log(self.tr("hymotion_msg_failed").format(returncode))
        self._current_action = None
        self.refresh_buttons()

    def run_generation(self):
        if not self.backend_dir.exists():
            messagebox.showwarning(self.tr("status_error"), self.tr("hymotion_msg_not_installed"))
            return
        prompt = self.prompt_text.get("1.0", "end").strip()
        if not prompt:
            messagebox.showwarning(self.tr("status_error"), self.tr("hymotion_msg_missing_prompt"))
            return
        model_key = self._model_options.get(self.model_var.get(), "HY-Motion-1.0-Lite")
        model_path = self.backend_dir / "ckpts" / "tencent" / model_key
        if not model_path.exists():
            messagebox.showwarning(self.tr("status_error"), self.tr("hymotion_msg_missing_weights").format(model_key))
            return
        output_dir = Path(self.output_entry.get().strip() or str(self.output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)

        temp_dir = self.app_root / "system" / "data" / "hymotion"
        temp_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = temp_dir / f"prompt_{int(time.time() * 1000)}.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        python_path = sys.executable.replace("pythonw.exe", "python.exe")
        self.log(self.tr("hymotion_msg_running"))
        self.set_busy(True)
        self._current_action = "run"
        self.run_process(
            [
                python_path,
                str(self.backend_dir / "local_infer.py"),
                "--model_path",
                str(model_path),
                "--input_text_dir",
                str(prompt_path.parent),
                "--output_dir",
                str(output_dir),
                "--disable_duration_est",
                "--disable_rewrite",
            ],
            on_done=self.on_run_done,
        )

    def on_run_done(self, returncode):
        self.set_busy(False)
        if returncode == 0:
            self.log(self.tr("hymotion_msg_run_done"))
        else:
            self.log(self.tr("hymotion_msg_failed").format(returncode))
        self._current_action = None
        self.refresh_buttons()

    def open_output_folder(self):
        output_dir = Path(self.output_entry.get().strip() or str(self.output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        os.startfile(str(output_dir))

    def resolve_backend_dir(self):
        primary = self.app_root / "system" / "ai-backends" / "HY-Motion-1.0"
        legacy = self.app_root.parent / "system" / "ai-backends" / "HY-Motion-1.0"
        if primary.exists():
            return primary
        if legacy.exists():
            return legacy
        return primary

    def detect_deps_installed(self):
        deps_marker = self.backend_dir / ".deps_installed"
        if deps_marker.exists():
            return True
        required = ["diffusers", "transformers", "torchdiffeq", "accelerate"]
        missing = []
        for pkg in required:
            try:
                importlib.metadata.version(pkg)
            except Exception:
                missing.append(pkg)
        if not missing:
            try:
                deps_marker.write_text("ok", encoding="utf-8")
            except Exception:
                pass
            return True
        return False
