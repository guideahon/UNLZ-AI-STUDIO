import customtkinter as ctk
import sys
import threading
import subprocess
import time
import logging
import importlib.metadata
from pathlib import Path
from tkinter import messagebox

from modules.base import StudioModule


class KleinModule(StudioModule):
    def __init__(self, parent):
        super().__init__(parent, "klein", "Flux 2 Klein")
        self.view = None
        self.app = parent

    def get_view(self) -> ctk.CTkFrame:
        self.view = KleinView(self.app.main_container, self.app)
        return self.view

    def on_enter(self):
        pass

    def on_leave(self):
        pass


class KleinView(ctk.CTkFrame):
    def __init__(self, master, app, **kwargs):
        super().__init__(master, **kwargs)
        self.app = app
        self.tr = app.tr
        self._busy = False
        self._current_action = None

        self.app_root = Path(__file__).resolve().parents[3]
        self.data_dir = self.app_root / "system" / "data" / "klein"
        self.output_dir = self.app_root / "system" / "klein-out"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._model_options = {
            self.tr("klein_model_4b"): "black-forest-labs/FLUX.2-klein-4B",
            self.tr("klein_model_4b_base"): "black-forest-labs/FLUX.2-klein-base-4B",
            self.tr("klein_model_9b"): "black-forest-labs/FLUX.2-klein-9B",
            self.tr("klein_model_9b_base"): "black-forest-labs/FLUX.2-klein-base-9B",
        }
        self.model_var = ctk.StringVar(value=self.tr("klein_model_4b"))

        self._device_options = {
            self.tr("klein_device_auto"): "auto",
            self.tr("klein_device_cuda"): "cuda",
            self.tr("klein_device_cpu"): "cpu",
        }
        self.device_var = ctk.StringVar(value=self.tr("klein_device_auto"))

        self.build_ui()
        self.refresh_buttons()

    def build_ui(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(10, 5))
        ctk.CTkLabel(header, text=self.tr("klein_title"), font=ctk.CTkFont(size=24, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(header, text=self.tr("klein_subtitle"), text_color="gray").pack(anchor="w")

        actions = ctk.CTkFrame(self, fg_color="transparent")
        actions.pack(fill="x", padx=10, pady=(10, 5))
        self.btn_deps = ctk.CTkButton(actions, text=self.tr("klein_btn_deps"), command=self.install_deps)
        self.btn_deps.pack(side="left", padx=5)
        self.btn_download = ctk.CTkButton(actions, text=self.tr("klein_btn_download_model"), command=self.download_model)
        self.btn_download.pack(side="left", padx=5)
        self.btn_open_output = ctk.CTkButton(actions, text=self.tr("klein_btn_open_output"), command=self.open_output_folder)
        self.btn_open_output.pack(side="left", padx=5)
        self.btn_open_repo = ctk.CTkButton(actions, text=self.tr("klein_btn_open_repo"), command=self.open_repo)
        self.btn_open_repo.pack(side="left", padx=5)

        status_frame = ctk.CTkFrame(self, fg_color="transparent")
        status_frame.pack(fill="x", padx=10, pady=(5, 0))
        ctk.CTkLabel(status_frame, text=self.tr("klein_status_label")).pack(side="left")
        self.status_value = ctk.CTkLabel(status_frame, text=self.tr("klein_status_idle"), text_color="gray")
        self.status_value.pack(side="left", padx=(6, 0))

        form = ctk.CTkFrame(self)
        form.pack(fill="both", expand=True, padx=10, pady=(10, 10))
        form.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(form, text=self.tr("klein_model_label")).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 4))
        self.model_menu = ctk.CTkOptionMenu(form, variable=self.model_var, values=list(self._model_options.keys()))
        self.model_menu.grid(row=0, column=1, sticky="ew", padx=10, pady=(10, 4))

        ctk.CTkLabel(form, text=self.tr("klein_device_label")).grid(row=1, column=0, sticky="w", padx=10, pady=(0, 4))
        self.device_menu = ctk.CTkOptionMenu(form, variable=self.device_var, values=list(self._device_options.keys()))
        self.device_menu.grid(row=1, column=1, sticky="ew", padx=10, pady=(0, 4))

        ctk.CTkLabel(form, text=self.tr("klein_prompt_label")).grid(row=2, column=0, columnspan=2, sticky="w", padx=10, pady=(10, 4))
        self.prompt_text = ctk.CTkTextbox(form, height=120)
        self.prompt_text.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=10, pady=(0, 10))

        size_frame = ctk.CTkFrame(form, fg_color="transparent")
        size_frame.grid(row=4, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 8))
        size_frame.grid_columnconfigure(1, weight=1)
        size_frame.grid_columnconfigure(3, weight=1)
        ctk.CTkLabel(size_frame, text=self.tr("klein_size_label")).grid(row=0, column=0, sticky="w", padx=(0, 8))
        ctk.CTkLabel(size_frame, text=self.tr("klein_width_label")).grid(row=0, column=1, sticky="w")
        self.width_entry = ctk.CTkEntry(size_frame)
        self.width_entry.insert(0, "1024")
        self.width_entry.grid(row=0, column=2, sticky="ew", padx=(6, 12))
        ctk.CTkLabel(size_frame, text=self.tr("klein_height_label")).grid(row=0, column=3, sticky="w")
        self.height_entry = ctk.CTkEntry(size_frame)
        self.height_entry.insert(0, "1024")
        self.height_entry.grid(row=0, column=4, sticky="ew", padx=(6, 0))

        params_frame = ctk.CTkFrame(form, fg_color="transparent")
        params_frame.grid(row=5, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 8))
        params_frame.grid_columnconfigure(1, weight=1)
        params_frame.grid_columnconfigure(3, weight=1)
        ctk.CTkLabel(params_frame, text=self.tr("klein_steps_label")).grid(row=0, column=0, sticky="w")
        self.steps_entry = ctk.CTkEntry(params_frame)
        self.steps_entry.insert(0, "4")
        self.steps_entry.grid(row=0, column=1, sticky="ew", padx=(6, 12))
        ctk.CTkLabel(params_frame, text=self.tr("klein_guidance_label")).grid(row=0, column=2, sticky="w")
        self.guidance_entry = ctk.CTkEntry(params_frame)
        self.guidance_entry.insert(0, "3.5")
        self.guidance_entry.grid(row=0, column=3, sticky="ew", padx=(6, 12))
        ctk.CTkLabel(params_frame, text=self.tr("klein_seed_label")).grid(row=0, column=4, sticky="w")
        self.seed_entry = ctk.CTkEntry(params_frame)
        self.seed_entry.grid(row=0, column=5, sticky="ew")

        output_frame = ctk.CTkFrame(form, fg_color="transparent")
        output_frame.grid(row=6, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
        output_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(output_frame, text=self.tr("klein_output_label")).grid(row=0, column=0, sticky="w")
        self.output_entry = ctk.CTkEntry(output_frame, placeholder_text=str(self.output_dir))
        self.output_entry.grid(row=0, column=1, sticky="ew", padx=(10, 0))

        buttons = ctk.CTkFrame(form, fg_color="transparent")
        buttons.grid(row=7, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 10))
        self.btn_generate = ctk.CTkButton(buttons, text=self.tr("klein_btn_generate"), command=self.run_generation)
        self.btn_generate.pack(side="left", padx=(0, 8))

        note = ctk.CTkLabel(self, text=self.tr("klein_note"), text_color="gray", wraplength=720, justify="left")
        note.pack(fill="x", padx=15, pady=(0, 10))

    def refresh_buttons(self):
        if self._busy:
            for btn in (self.btn_deps, self.btn_download, self.btn_open_output, self.btn_open_repo, self.btn_generate):
                btn.configure(state="disabled")
            return

        deps_ok = self.check_deps_available()
        self.btn_deps.configure(text=self.tr("klein_btn_deps_installed") if deps_ok else self.tr("klein_btn_deps"))
        for btn in (self.btn_deps, self.btn_download, self.btn_open_output, self.btn_open_repo, self.btn_generate):
            btn.configure(state="normal")

    def set_busy(self, busy):
        self._busy = busy
        if self.status_value:
            self.status_value.configure(text=self.tr("klein_status_busy") if busy else self.tr("klein_status_idle"))
        self.refresh_buttons()

    def log(self, message):
        logging.info(message)

    def safe_log(self, message):
        self.after(0, lambda: logging.info(message))

    def check_deps_available(self):
        required = ["diffusers", "transformers", "accelerate", "safetensors", "huggingface_hub", "torch", "PIL"]
        for pkg in required:
            try:
                if pkg == "PIL":
                    import PIL  # noqa: F401
                else:
                    importlib.metadata.version(pkg)
            except Exception:
                return False
        return True

    def install_deps(self):
        python_path = sys.executable.replace("pythonw.exe", "python.exe")
        packages = ["diffusers", "transformers", "accelerate", "safetensors", "huggingface_hub", "pillow"]
        self.log(self.tr("klein_msg_installing_deps"))
        self.set_busy(True)
        self._current_action = "deps"
        self.run_process([python_path, "-m", "pip", "install", *packages], on_done=self.on_process_done)

    def download_model(self):
        if not self.check_deps_available():
            messagebox.showwarning(self.tr("status_error"), self.tr("klein_msg_missing_deps"))
            return
        script_path = self.data_dir / "klein_run.py"
        if not script_path.exists():
            messagebox.showwarning(self.tr("status_error"), self.tr("klein_msg_missing_script"))
            return
        model_id = self._model_options.get(self.model_var.get())
        self.log(self.tr("klein_msg_downloading_model"))
        self.set_busy(True)
        self._current_action = "download"
        python_path = sys.executable.replace("pythonw.exe", "python.exe")
        self.run_process(
            [python_path, "-u", str(script_path), "--model", model_id, "--download-only"],
            on_done=self.on_download_done,
        )

    def run_generation(self):
        if not self.check_deps_available():
            messagebox.showwarning(self.tr("status_error"), self.tr("klein_msg_missing_deps"))
            return
        prompt = self.prompt_text.get("1.0", "end").strip()
        if not prompt:
            messagebox.showwarning(self.tr("status_error"), self.tr("klein_msg_missing_prompt"))
            return
        script_path = self.data_dir / "klein_run.py"
        if not script_path.exists():
            messagebox.showwarning(self.tr("status_error"), self.tr("klein_msg_missing_script"))
            return

        width = self.parse_int(self.width_entry.get().strip() or "1024", self.tr("klein_width_label"))
        height = self.parse_int(self.height_entry.get().strip() or "1024", self.tr("klein_height_label"))
        steps = self.parse_int(self.steps_entry.get().strip() or "4", self.tr("klein_steps_label"))
        guidance = self.parse_float(self.guidance_entry.get().strip() or "3.5", self.tr("klein_guidance_label"))
        if width is None or height is None or steps is None or guidance is None:
            return
        seed = self.seed_entry.get().strip()
        seed_val = None
        if seed:
            seed_val = self.parse_int(seed, self.tr("klein_seed_label"))
            if seed_val is None:
                return

        output_dir = Path(self.output_entry.get().strip() or str(self.output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"klein_{int(time.time() * 1000)}.png"

        model_id = self._model_options.get(self.model_var.get())
        device = self._device_options.get(self.device_var.get(), "auto")
        python_path = sys.executable.replace("pythonw.exe", "python.exe")
        cmd = [
            python_path,
            "-u",
            str(script_path),
            "--model",
            model_id,
            "--prompt",
            prompt,
            "--width",
            str(width),
            "--height",
            str(height),
            "--steps",
            str(steps),
            "--guidance",
            str(guidance),
            "--output",
            str(output_path),
            "--device",
            device,
        ]
        if seed_val is not None:
            cmd.extend(["--seed", str(seed_val)])

        self.set_busy(True)
        self._current_action = "run"
        self.run_process(cmd, on_done=lambda code: self.on_generation_done(code, output_path))

    def parse_int(self, value, label_key):
        try:
            return int(value)
        except Exception:
            messagebox.showwarning(self.tr("status_error"), self.tr("klein_msg_invalid_number").format(label_key))
            return None

    def parse_float(self, value, label_key):
        try:
            return float(value)
        except Exception:
            messagebox.showwarning(self.tr("status_error"), self.tr("klein_msg_invalid_number").format(label_key))
            return None

    def run_process(self, cmd, on_done=None):
        def worker():
            returncode = 1
            try:
                process = subprocess.Popen(
                    cmd,
                    cwd=str(self.data_dir),
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
                if returncode == 0 and self._current_action == "deps":
                    self.safe_log(self.tr("klein_msg_deps_done"))
                elif returncode == 0:
                    self.safe_log(self.tr("klein_msg_done"))
                else:
                    self.safe_log(self.tr("klein_msg_failed").format(returncode))
            except Exception as exc:
                self.safe_log(f"{self.tr('status_error')}: {exc}")
            if on_done:
                self.after(0, lambda: on_done(returncode))
            else:
                self.after(0, lambda: self.on_process_done(returncode))

        threading.Thread(target=worker, daemon=True).start()

    def on_process_done(self, returncode):
        self.set_busy(False)
        if returncode != 0:
            self.log(self.tr("klein_msg_failed").format(returncode))
        self._current_action = None
        self.refresh_buttons()

    def on_download_done(self, returncode):
        self.set_busy(False)
        if returncode == 0:
            self.log(self.tr("klein_msg_download_done"))
        else:
            self.log(self.tr("klein_msg_failed").format(returncode))
        self._current_action = None
        self.refresh_buttons()

    def on_generation_done(self, returncode, output_path):
        self.set_busy(False)
        if returncode == 0:
            self.log(self.tr("klein_msg_output_saved").format(output_path))
        else:
            self.log(self.tr("klein_msg_failed").format(returncode))
        self._current_action = None
        self.refresh_buttons()

    def open_output_folder(self):
        output_dir = Path(self.output_entry.get().strip() or str(self.output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            import os
            os.startfile(str(output_dir))
        except Exception as exc:
            messagebox.showwarning(self.tr("status_error"), f"{self.tr('status_error')}: {exc}")

    def open_repo(self):
        import webbrowser
        webbrowser.open("https://huggingface.co/black-forest-labs/FLUX.2-klein-4B")
