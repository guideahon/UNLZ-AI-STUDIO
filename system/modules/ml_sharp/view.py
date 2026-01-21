import customtkinter as ctk
import os
import sys
import threading
import subprocess
import shutil
from pathlib import Path
from tkinter import filedialog, messagebox

from modules.base import StudioModule


class MLSharpModule(StudioModule):
    def __init__(self, parent):
        super().__init__(parent, "ml_sharp", "ML-SHARP")
        self.view = None
        self.app = parent

    def get_view(self) -> ctk.CTkFrame:
        self.view = MLSharpView(self.app.main_container, self.app)
        return self.view

    def on_enter(self):
        pass

    def on_leave(self):
        pass


class MLSharpView(ctk.CTkFrame):
    def __init__(self, master, app, **kwargs):
        super().__init__(master, **kwargs)
        self.app = app
        self.tr = app.tr
        self._busy = False

        self.app_root = Path(__file__).resolve().parents[3]
        self.backend_dir = self.app_root / "system" / "ai-backends" / "ml-sharp"
        self.default_output_dir = self.app_root / "system" / "ml-sharp-out"
        self.deps_marker = self.backend_dir / ".deps_installed"

        self.build_ui()
        self.refresh_buttons()

    def build_ui(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(10, 5))

        ctk.CTkLabel(header, text=self.tr("mlsharp_title"), font=ctk.CTkFont(size=24, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(header, text=self.tr("mlsharp_subtitle"), text_color="gray").pack(anchor="w")

        actions = ctk.CTkFrame(self, fg_color="transparent")
        actions.pack(fill="x", padx=10, pady=(10, 5))

        self.btn_deps = ctk.CTkButton(actions, text=self.tr("mlsharp_btn_deps"), command=self.install_deps)
        self.btn_deps.pack(side="left", padx=5)

        self.btn_open = ctk.CTkButton(actions, text=self.tr("mlsharp_btn_open_folder"), command=self.open_backend_folder)
        self.btn_open.pack(side="left", padx=5)

        self.btn_repo = ctk.CTkButton(actions, text=self.tr("mlsharp_btn_open_repo"), command=self.open_repo)
        self.btn_repo.pack(side="left", padx=5)

        status_frame = ctk.CTkFrame(self, fg_color="transparent")
        status_frame.pack(fill="x", padx=10, pady=(5, 0))
        ctk.CTkLabel(status_frame, text=self.tr("mlsharp_status_label")).pack(side="left")
        self.status_value = ctk.CTkLabel(status_frame, text=self.tr("mlsharp_status_idle"), text_color="gray")
        self.status_value.pack(side="left", padx=(6, 0))

        input_frame = ctk.CTkFrame(self)
        input_frame.pack(fill="x", padx=10, pady=(10, 5))
        input_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(input_frame, text=self.tr("mlsharp_input_label")).grid(row=0, column=0, padx=10, pady=8, sticky="w")
        self.input_entry = ctk.CTkEntry(input_frame, placeholder_text=self.tr("mlsharp_input_placeholder"))
        self.input_entry.grid(row=0, column=1, padx=10, pady=8, sticky="ew")

        ctk.CTkButton(input_frame, text=self.tr("mlsharp_btn_browse_file"), command=self.browse_input_file).grid(row=0, column=2, padx=6, pady=8)
        ctk.CTkButton(input_frame, text=self.tr("mlsharp_btn_browse_folder"), command=self.browse_input_folder).grid(row=0, column=3, padx=6, pady=8)

        output_frame = ctk.CTkFrame(self)
        output_frame.pack(fill="x", padx=10, pady=(5, 5))
        output_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(output_frame, text=self.tr("mlsharp_output_label")).grid(row=0, column=0, padx=10, pady=8, sticky="w")
        self.output_entry = ctk.CTkEntry(output_frame, placeholder_text=self.tr("mlsharp_output_placeholder"))
        self.output_entry.grid(row=0, column=1, padx=10, pady=8, sticky="ew")
        ctk.CTkButton(output_frame, text=self.tr("mlsharp_btn_browse_output"), command=self.browse_output_folder).grid(row=0, column=2, padx=6, pady=8)

        options_frame = ctk.CTkFrame(self)
        options_frame.pack(fill="x", padx=10, pady=(5, 5))

        ctk.CTkLabel(options_frame, text=self.tr("mlsharp_device_label")).pack(side="left", padx=10)
        self.device_var = ctk.StringVar(value=self.tr("mlsharp_device_default"))
        self.device_menu = ctk.CTkOptionMenu(
            options_frame,
            variable=self.device_var,
            values=[
                self.tr("mlsharp_device_default"),
                self.tr("mlsharp_device_cpu"),
                self.tr("mlsharp_device_cuda"),
                self.tr("mlsharp_device_mps"),
            ],
        )
        self.device_menu.pack(side="left", padx=5)

        self.render_var = ctk.BooleanVar(value=False)
        self.render_check = ctk.CTkCheckBox(options_frame, text=self.tr("mlsharp_render_label"), variable=self.render_var)
        self.render_check.pack(side="left", padx=15)

        controls = ctk.CTkFrame(self, fg_color="transparent")
        controls.pack(fill="x", padx=10, pady=(5, 5))

        self.btn_run = ctk.CTkButton(controls, text=self.tr("mlsharp_btn_run"), command=self.run_predict)
        self.btn_run.pack(side="left", padx=5)

        self.btn_open_output = ctk.CTkButton(controls, text=self.tr("mlsharp_btn_open_output"), command=self.open_output_folder)
        self.btn_open_output.pack(side="left", padx=5)

        log_frame = ctk.CTkFrame(self)
        log_frame.pack(fill="both", expand=True, padx=10, pady=(5, 10))

        ctk.CTkLabel(log_frame, text=self.tr("mlsharp_log_label")).pack(anchor="w", padx=10, pady=(10, 0))
        self.log_textbox = ctk.CTkTextbox(log_frame, font=ctk.CTkFont(family="Consolas", size=11))
        self.log_textbox.pack(fill="both", expand=True, padx=10, pady=10)
        self.log_textbox.configure(state="disabled")

    def refresh_buttons(self):
        disabled = "disabled" if self._busy else "normal"
        for btn in (self.btn_deps, self.btn_open, self.btn_repo, self.btn_run, self.btn_open_output):
            btn.configure(state=disabled)

        if self._busy:
            return

        if not self.backend_dir.exists():
            self.btn_deps.configure(state="disabled")
            self.btn_open.configure(state="disabled")
        else:
            if self.deps_marker.exists():
                self.btn_deps.configure(text=self.tr("mlsharp_btn_deps_installed"))
            else:
                self.btn_deps.configure(text=self.tr("mlsharp_btn_deps"))

    def set_busy(self, busy):
        self._busy = busy
        self.status_value.configure(text=self.tr("mlsharp_status_busy") if busy else self.tr("mlsharp_status_idle"))
        self.refresh_buttons()

    def browse_input_file(self):
        file_path = filedialog.askopenfilename(
            title=self.tr("mlsharp_dialog_input"),
            filetypes=[
                ("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.webp"),
                ("All files", "*.*"),
            ],
        )
        if file_path:
            self.input_entry.delete(0, "end")
            self.input_entry.insert(0, file_path)

    def browse_input_folder(self):
        folder_path = filedialog.askdirectory(title=self.tr("mlsharp_dialog_input_folder"))
        if folder_path:
            self.input_entry.delete(0, "end")
            self.input_entry.insert(0, folder_path)

    def browse_output_folder(self):
        folder_path = filedialog.askdirectory(title=self.tr("mlsharp_dialog_output"))
        if folder_path:
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, folder_path)

    def open_backend_folder(self):
        if self.backend_dir.exists():
            os.startfile(str(self.backend_dir))

    def open_repo(self):
        import webbrowser
        webbrowser.open("https://github.com/apple/ml-sharp")

    def open_output_folder(self):
        output_dir = self.get_output_dir()
        if output_dir is None or not output_dir.exists():
            messagebox.showwarning(self.tr("status_error"), self.tr("mlsharp_msg_output_missing"))
            return
        os.startfile(str(output_dir))

    def install_deps(self):
        if not self.backend_dir.exists():
            messagebox.showwarning(self.tr("status_error"), self.tr("mlsharp_msg_backend_missing"))
            return
        req_path = self.backend_dir / "requirements.txt"
        if not req_path.exists():
            messagebox.showwarning(self.tr("status_error"), self.tr("mlsharp_msg_requirements_missing"))
            return
        python_path = sys.executable.replace("pythonw.exe", "python.exe")
        self.log(self.tr("mlsharp_msg_deps_installing"))
        self.set_busy(True)

        def worker():
            code = self.run_command([python_path, "-m", "pip", "install", "-r", str(req_path)])
            if code == 0:
                code = self.run_command([python_path, "-m", "pip", "install", "-e", str(self.backend_dir)])
            self.after(0, lambda: self.on_deps_done(code))

        threading.Thread(target=worker, daemon=True).start()

    def on_deps_done(self, returncode):
        self.set_busy(False)
        if returncode == 0:
            self.log(self.tr("mlsharp_msg_deps_done"))
            try:
                self.deps_marker.write_text("ok", encoding="utf-8")
            except Exception:
                pass
        else:
            self.log(self.tr("mlsharp_msg_failed").format(returncode))
        self.refresh_buttons()

    def run_predict(self):
        input_path = self.input_entry.get().strip()
        if not input_path or not os.path.exists(input_path):
            messagebox.showwarning(self.tr("status_error"), self.tr("mlsharp_msg_missing_input"))
            return
        if not self.backend_dir.exists():
            messagebox.showwarning(self.tr("status_error"), self.tr("mlsharp_msg_backend_missing"))
            return

        output_dir = self.get_output_dir()
        if output_dir is None:
            messagebox.showwarning(self.tr("status_error"), self.tr("mlsharp_msg_output_missing"))
            return
        output_dir.mkdir(parents=True, exist_ok=True)

        sharp_cmd = self.find_sharp_cmd()
        if not sharp_cmd:
            messagebox.showwarning(self.tr("status_error"), self.tr("mlsharp_msg_sharp_missing"))
            return

        cmd = [sharp_cmd, "predict", "-i", input_path, "-o", str(output_dir)]
        if self.render_var.get():
            cmd.append("--render")

        device = self.map_device_value(self.device_var.get())
        if device != "default":
            cmd.extend(["--device", device])

        self.log(self.tr("mlsharp_msg_running"))
        self.set_busy(True)

        threading.Thread(target=lambda: self.run_predict_worker(cmd), daemon=True).start()

    def run_predict_worker(self, cmd):
        returncode = self.run_command(cmd)
        self.after(0, lambda: self.on_predict_done(returncode))

    def on_predict_done(self, returncode):
        self.set_busy(False)
        if returncode == 0:
            self.log(self.tr("mlsharp_msg_done"))
        else:
            self.log(self.tr("mlsharp_msg_failed").format(returncode))

    def run_command(self, cmd):
        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(self.backend_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
            )
            for line in process.stdout:
                self.safe_log(line.rstrip())
            process.wait()
            return process.returncode
        except Exception as exc:
            self.safe_log(f"{self.tr('status_error')}: {exc}")
            return 1

    def safe_log(self, message):
        self.after(0, lambda: self.log(message))

    def log(self, message):
        self.log_textbox.configure(state="normal")
        self.log_textbox.insert("end", message + "\n")
        self.log_textbox.see("end")
        self.log_textbox.configure(state="disabled")

    def get_output_dir(self):
        output_path = self.output_entry.get().strip()
        if output_path:
            return Path(output_path)
        return self.default_output_dir

    def find_sharp_cmd(self):
        scripts_dir = Path(sys.executable).resolve().parent / "Scripts"
        for name in ("sharp.exe", "sharp"):
            candidate = scripts_dir / name
            if candidate.exists():
                return str(candidate)
        return shutil.which("sharp")

    def map_device_value(self, value):
        mapping = {
            self.tr("mlsharp_device_default"): "default",
            self.tr("mlsharp_device_cpu"): "cpu",
            self.tr("mlsharp_device_cuda"): "cuda",
            self.tr("mlsharp_device_mps"): "mps",
        }
        return mapping.get(value, "default")
