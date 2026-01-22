import customtkinter as ctk
import os
import sys
import threading
import subprocess
import time
import webbrowser
import logging
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog

from modules.base import StudioModule


class NeuttsModule(StudioModule):
    def __init__(self, parent):
        super().__init__(parent, "neutts", "NeuTTS")
        self.view = None
        self.app = parent

    def get_view(self) -> ctk.CTkFrame:
        self.view = NeuttsView(self.app.main_container, self.app)
        return self.view

    def on_enter(self):
        pass

    def on_leave(self):
        pass


class NeuttsView(ctk.CTkFrame):
    def __init__(self, master, app, **kwargs):
        super().__init__(master, **kwargs)
        self.app = app
        self.tr = app.tr
        self._busy = False
        self._current_action = None
        self._last_output = None
        self.current_repo = ""

        self.app_root = Path(__file__).resolve().parents[3]
        self.backend_dir = self.app_root / "system" / "ai-backends" / "neutts"
        self.data_dir = self.app_root / "system" / "data" / "neutts"
        self.output_dir = self.app_root / "system" / "neutts-out"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model_groups = {
            "air": {
                "label": self.tr("neutts_model_air"),
                "variants": [
                    (self.tr("neutts_variant_full"), "neuphonic/neutts-air"),
                    (self.tr("neutts_variant_q8"), "neuphonic/neutts-air-q8-gguf"),
                    (self.tr("neutts_variant_q4"), "neuphonic/neutts-air-q4-gguf"),
                ],
                "space": "https://huggingface.co/spaces/neuphonic/neutts-air",
            },
            "nano": {
                "label": self.tr("neutts_model_nano"),
                "variants": [
                    (self.tr("neutts_variant_full"), "neuphonic/neutts-nano"),
                    (self.tr("neutts_variant_q8"), "neuphonic/neutts-nano-q8-gguf"),
                    (self.tr("neutts_variant_q4"), "neuphonic/neutts-nano-q4-gguf"),
                ],
                "space": "https://huggingface.co/spaces/neuphonic/neutts-nano",
            },
        }

        self.model_group_labels = [self.model_groups["air"]["label"], self.model_groups["nano"]["label"]]
        self.model_group_map = {
            self.model_groups["air"]["label"]: "air",
            self.model_groups["nano"]["label"]: "nano",
        }

        self.build_ui()
        self.refresh_buttons()

    def build_ui(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(10, 5))
        ctk.CTkLabel(header, text=self.tr("neutts_title"), font=ctk.CTkFont(size=24, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(header, text=self.tr("neutts_subtitle"), text_color="gray").pack(anchor="w")

        actions = ctk.CTkFrame(self, fg_color="transparent")
        actions.pack(fill="x", padx=10, pady=(10, 5))
        self.btn_install = ctk.CTkButton(actions, text=self.tr("neutts_btn_install"), command=self.install_backend)
        self.btn_install.pack(side="left", padx=5)
        self.btn_uninstall = ctk.CTkButton(actions, text=self.tr("neutts_btn_uninstall"), command=self.uninstall_backend)
        self.btn_uninstall.pack(side="left", padx=5)
        self.btn_deps = ctk.CTkButton(actions, text=self.tr("neutts_btn_deps"), command=self.install_deps)
        self.btn_deps.pack(side="left", padx=5)
        self.btn_open = ctk.CTkButton(actions, text=self.tr("neutts_btn_open_folder"), command=self.open_backend_folder)
        self.btn_open.pack(side="left", padx=5)
        self.btn_repo = ctk.CTkButton(actions, text=self.tr("neutts_btn_open_repo"), command=self.open_repo)
        self.btn_repo.pack(side="left", padx=5)
        self.btn_espeak = ctk.CTkButton(actions, text=self.tr("neutts_btn_open_espeak"), command=self.open_espeak)
        self.btn_espeak.pack(side="left", padx=5)
        self.btn_detect_espeak = ctk.CTkButton(actions, text=self.tr("neutts_btn_detect_espeak"), command=self.detect_espeak)
        self.btn_detect_espeak.pack(side="left", padx=5)

        status_frame = ctk.CTkFrame(self, fg_color="transparent")
        status_frame.pack(fill="x", padx=10, pady=(5, 0))
        ctk.CTkLabel(status_frame, text=self.tr("neutts_status_label")).pack(side="left")
        self.status_value = ctk.CTkLabel(status_frame, text=self.tr("neutts_status_idle"), text_color="gray")
        self.status_value.pack(side="left", padx=(6, 0))
        self.espeak_status_label = ctk.CTkLabel(status_frame, text="", text_color="gray")
        self.espeak_status_label.pack(side="left", padx=(12, 0))

        model_frame = ctk.CTkFrame(self)
        model_frame.pack(fill="x", padx=10, pady=(10, 5))
        ctk.CTkLabel(model_frame, text=self.tr("neutts_model_label")).grid(row=0, column=0, sticky="w", padx=10, pady=6)
        self.model_group = ctk.CTkSegmentedButton(model_frame, values=self.model_group_labels, command=self.on_model_group_change)
        self.model_group.grid(row=0, column=1, sticky="w", padx=10, pady=6)
        self.model_group.set(self.model_group_labels[0])

        ctk.CTkLabel(model_frame, text=self.tr("neutts_variant_label")).grid(row=1, column=0, sticky="w", padx=10, pady=6)
        self.variant_var = ctk.StringVar(value="")
        self.variant_menu = ctk.CTkOptionMenu(model_frame, variable=self.variant_var, values=[""], command=self.on_variant_change)
        self.variant_menu.grid(row=1, column=1, sticky="w", padx=10, pady=6)

        ctk.CTkLabel(model_frame, text=self.tr("neutts_device_label")).grid(row=2, column=0, sticky="w", padx=10, pady=6)
        self.device_var = ctk.StringVar(value="cpu")
        self.device_menu = ctk.CTkOptionMenu(model_frame, variable=self.device_var, values=["cpu", "cuda"])
        self.device_menu.grid(row=2, column=1, sticky="w", padx=10, pady=6)

        input_frame = ctk.CTkFrame(self)
        input_frame.pack(fill="both", expand=True, padx=10, pady=(10, 5))
        input_frame.grid_columnconfigure(1, weight=1)
        input_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(input_frame, text=self.tr("neutts_input_text_label")).grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(10, 5))
        self.input_text = ctk.CTkTextbox(input_frame, height=120)
        self.input_text.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=(0, 10))

        ctk.CTkLabel(input_frame, text=self.tr("neutts_ref_audio_label")).grid(row=2, column=0, sticky="w", padx=10, pady=6)
        self.ref_audio_var = ctk.StringVar(value="")
        self.ref_audio_entry = ctk.CTkEntry(input_frame, textvariable=self.ref_audio_var)
        self.ref_audio_entry.grid(row=2, column=1, sticky="ew", padx=10, pady=6)
        self.btn_ref_audio = ctk.CTkButton(input_frame, text=self.tr("neutts_btn_browse"), command=self.select_ref_audio, width=100)
        self.btn_ref_audio.grid(row=2, column=2, sticky="w", padx=(0, 10), pady=6)
        self.btn_ref_record = ctk.CTkButton(input_frame, text=self.tr("neutts_btn_record"), command=self.record_ref_audio, width=120)
        self.btn_ref_record.grid(row=2, column=3, sticky="w", padx=(0, 10), pady=6)

        ctk.CTkLabel(input_frame, text=self.tr("neutts_ref_text_label")).grid(row=3, column=0, sticky="w", padx=10, pady=6)
        self.ref_text = ctk.CTkTextbox(input_frame, height=80)
        self.ref_text.grid(row=3, column=1, columnspan=2, sticky="nsew", padx=10, pady=6)
        self.btn_ref_text = ctk.CTkButton(input_frame, text=self.tr("neutts_btn_load_text"), command=self.load_ref_text, width=120)
        self.btn_ref_text.grid(row=4, column=2, sticky="e", padx=(0, 10), pady=(0, 10))

        output_frame = ctk.CTkFrame(self)
        output_frame.pack(fill="x", padx=10, pady=(10, 10))
        output_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(output_frame, text=self.tr("neutts_output_label")).grid(row=0, column=0, sticky="w", padx=10, pady=6)
        self.output_path_label = ctk.CTkLabel(output_frame, text=self.tr("neutts_output_none"), text_color="gray")
        self.output_path_label.grid(row=0, column=1, sticky="w", padx=10, pady=6)

        buttons = ctk.CTkFrame(output_frame, fg_color="transparent")
        buttons.grid(row=1, column=0, columnspan=2, sticky="w", padx=10, pady=(5, 10))
        self.btn_generate = ctk.CTkButton(buttons, text=self.tr("neutts_btn_generate"), command=self.generate_audio)
        self.btn_generate.pack(side="left", padx=5)
        self.btn_preview = ctk.CTkButton(buttons, text=self.tr("neutts_btn_preview"), command=self.preview_audio)
        self.btn_preview.pack(side="left", padx=5)
        self.btn_stop_audio = ctk.CTkButton(buttons, text=self.tr("neutts_btn_stop"), command=self.stop_preview)
        self.btn_stop_audio.pack(side="left", padx=5)
        self.btn_save = ctk.CTkButton(buttons, text=self.tr("neutts_btn_save"), command=self.save_output)
        self.btn_save.pack(side="left", padx=5)
        self.btn_open_output = ctk.CTkButton(buttons, text=self.tr("neutts_btn_open_output"), command=self.open_output_folder)
        self.btn_open_output.pack(side="left", padx=5)

        note = ctk.CTkLabel(self, text=self.tr("neutts_note"), text_color="gray", wraplength=700, justify="left")
        note.pack(fill="x", padx=15, pady=(0, 10))

        self.on_model_group_change(self.model_group.get())
        self.update_espeak_status()

    def set_busy(self, busy):
        self._busy = busy
        if self.status_value:
            self.status_value.configure(
                text=self.tr("status_in_progress") if busy else self.tr("neutts_status_idle")
            )
        self.refresh_buttons()

    def refresh_buttons(self):
        if self._busy:
            for btn in (
                self.btn_install, self.btn_uninstall, self.btn_deps, self.btn_open, self.btn_repo, self.btn_espeak,
                self.btn_detect_espeak,
                self.btn_generate, self.btn_preview, self.btn_stop_audio, self.btn_save, self.btn_open_output,
                self.btn_ref_audio, self.btn_ref_record, self.btn_ref_text
            ):
                btn.configure(state="disabled")
            return

        installed = self.backend_dir.exists()
        deps_ok = self.check_deps_available()
        if installed:
            self.btn_install.pack_forget()
            if not self.btn_uninstall.winfo_manager():
                self.btn_uninstall.pack(side="left", padx=5)
            if deps_ok:
                self.btn_deps.pack_forget()
            elif not self.btn_deps.winfo_manager():
                self.btn_deps.pack(side="left", padx=5)
            if not self.btn_open.winfo_manager():
                self.btn_open.pack(side="left", padx=5)
        else:
            self.btn_uninstall.pack_forget()
            self.btn_deps.pack_forget()
            self.btn_open.pack_forget()
            if not self.btn_install.winfo_manager():
                self.btn_install.pack(side="left", padx=5)

        espeak_ok, _ = self.check_espeak_available()
        if espeak_ok:
            self.btn_espeak.pack_forget()
        elif not self.btn_espeak.winfo_manager():
            self.btn_espeak.pack(side="left", padx=5)

    def check_deps_available(self):
        try:
            import neutts  # noqa: F401
            import soundfile  # noqa: F401
        except Exception:
            return False
        needs_gguf = bool(self.current_repo and self.current_repo.endswith("gguf"))
        if needs_gguf:
            try:
                import llama_cpp  # noqa: F401
            except Exception:
                return False
        return True

        for btn in (
            self.btn_install, self.btn_uninstall, self.btn_deps, self.btn_open, self.btn_repo, self.btn_espeak,
            self.btn_detect_espeak,
            self.btn_generate, self.btn_preview, self.btn_stop_audio, self.btn_save, self.btn_open_output,
            self.btn_ref_audio, self.btn_ref_record, self.btn_ref_text
        ):
            btn.configure(state="normal")

    def on_model_group_change(self, value=None):
        selected = value or self.model_group.get()
        key = self.model_group_map.get(selected, "air")
        variants = self.model_groups[key]["variants"]
        labels = [v[0] for v in variants]
        self.variant_menu.configure(values=labels)
        if labels:
            self.variant_var.set(labels[0])
        self.on_variant_change(labels[0] if labels else "")

    def on_variant_change(self, value=None):
        selected_group = self.model_group.get()
        group_key = self.model_group_map.get(selected_group, "air")
        variants = self.model_groups[group_key]["variants"]
        label = value or self.variant_var.get()
        repo = ""
        for variant_label, repo_id in variants:
            if variant_label == label:
                repo = repo_id
                break
        self.current_repo = repo

    def install_backend(self):
        if self.backend_dir.exists():
            logging.info(self.tr("neutts_msg_already_installed"))
            self.refresh_buttons()
            return
        if not self.check_git_available():
            messagebox.showwarning(self.tr("status_error"), self.tr("neutts_msg_git_missing"))
            return
        self.backend_dir.parent.mkdir(parents=True, exist_ok=True)
        self.set_busy(True)
        self._current_action = "install"
        self.run_process(["git", "clone", "--depth", "1", "https://github.com/neuphonic/neutts", str(self.backend_dir)])

    def uninstall_backend(self):
        if not self.backend_dir.exists():
            logging.info(self.tr("neutts_msg_not_installed"))
            self.refresh_buttons()
            return
        try:
            import shutil
            shutil.rmtree(self.backend_dir)
            logging.info(self.tr("neutts_msg_uninstalled"))
        except Exception as exc:
            logging.info(f"{self.tr('status_error')}: {exc}")
        self.refresh_buttons()

    def install_deps(self):
        if not self.backend_dir.exists():
            messagebox.showwarning(self.tr("status_error"), self.tr("neutts_msg_not_installed"))
            return
        req_path = self.backend_dir / "requirements.txt"
        if not req_path.exists():
            messagebox.showwarning(self.tr("status_error"), self.tr("neutts_msg_requirements_missing"))
            return
        python_path = sys.executable.replace("pythonw.exe", "python.exe")
        packages = ["-r", str(req_path), "soundfile"]
        if self.current_repo and self.current_repo.endswith("gguf"):
            packages.append("llama-cpp-python")
        self.set_busy(True)
        self._current_action = "deps"
        self.run_process([python_path, "-m", "pip", "install", *packages])

    def open_backend_folder(self):
        if self.backend_dir.exists():
            os.startfile(str(self.backend_dir))

    def open_repo(self):
        webbrowser.open("https://github.com/neuphonic/neutts")

    def open_espeak(self):
        webbrowser.open("https://github.com/espeak-ng/espeak-ng/releases")

    def detect_espeak(self):
        ok, detail = self.check_espeak_available()
        if ok:
            messagebox.showinfo(self.tr("neutts_btn_detect_espeak"), self.tr("neutts_msg_espeak_found").format(detail))
        else:
            messagebox.showwarning(self.tr("neutts_btn_detect_espeak"), self.tr("neutts_msg_espeak_missing"))
        self.update_espeak_status()

    def check_espeak_available(self):
        from shutil import which
        exe = which("espeak-ng") or which("espeak")
        if exe:
            return True, exe
        lib_path = os.environ.get("PHONEMIZER_ESPEAK_LIBRARY", "")
        bin_path = os.environ.get("PHONEMIZER_ESPEAK_PATH", "")
        if lib_path and Path(lib_path).exists():
            return True, lib_path
        if bin_path and Path(bin_path).exists():
            return True, bin_path
        return False, ""

    def update_espeak_status(self):
        ok, detail = self.check_espeak_available()
        if ok:
            text = self.tr("neutts_espeak_status_ok_path").format(detail)
            color = "green"
        else:
            text = self.tr("neutts_espeak_status_missing")
            color = "orange"
        self.espeak_status_label.configure(text=text, text_color=color)

    def check_git_available(self):
        try:
            result = subprocess.run(["git", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return result.returncode == 0
        except Exception:
            return False

    def run_process(self, cmd):
        def worker():
            returncode = 1
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
                    logging.info(line.rstrip())
                process.wait()
                returncode = process.returncode
            except Exception as exc:
                logging.info(f"{self.tr('status_error')}: {exc}")
            self.after(0, lambda: self.on_process_done(returncode))

        threading.Thread(target=worker, daemon=True).start()

    def on_process_done(self, returncode):
        self.set_busy(False)
        if returncode == 0:
            logging.info(self.tr("neutts_msg_done"))
        else:
            logging.info(self.tr("neutts_msg_failed").format(returncode))
        self._current_action = None
        self.refresh_buttons()

    def select_ref_audio(self):
        file_path = filedialog.askopenfilename(
            title=self.tr("neutts_ref_audio_label"),
            filetypes=[("WAV", "*.wav"), ("All files", "*.*")],
        )
        if file_path:
            self.ref_audio_var.set(file_path)

    def load_ref_text(self):
        file_path = filedialog.askopenfilename(
            title=self.tr("neutts_ref_text_label"),
            filetypes=[("Text", "*.txt"), ("All files", "*.*")],
        )
        if not file_path:
            return
        try:
            content = Path(file_path).read_text(encoding="utf-8").strip()
            self.ref_text.delete("0.0", "end")
            self.ref_text.insert("end", content)
        except Exception as exc:
            messagebox.showwarning(self.tr("status_error"), f"{self.tr('status_error')}: {exc}")

    def record_ref_audio(self):
        duration = simpledialog.askinteger(
            self.tr("neutts_btn_record"),
            self.tr("neutts_record_prompt"),
            initialvalue=5,
            minvalue=2,
            maxvalue=30,
        )
        if not duration:
            return
        try:
            import sounddevice as sd
            import soundfile as sf
        except Exception:
            if messagebox.askyesno(self.tr("status_error"), self.tr("neutts_msg_record_deps")):
                self.install_python_packages(["sounddevice", "soundfile"])
            return

        def worker():
            try:
                self.set_busy(True)
                samplerate = 24000
                channels = 1
                data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype="float32")
                sd.wait()
                self.data_dir.mkdir(parents=True, exist_ok=True)
                output_path = self.data_dir / f"neutts_ref_{int(time.time() * 1000)}.wav"
                sf.write(str(output_path), data, samplerate)
                self.after(0, lambda: self.ref_audio_var.set(str(output_path)))
            except Exception as exc:
                self.after(0, lambda: messagebox.showwarning(self.tr("status_error"), f"{self.tr('status_error')}: {exc}"))
            finally:
                self.after(0, lambda: self.set_busy(False))

        threading.Thread(target=worker, daemon=True).start()

    def install_python_packages(self, packages):
        if not packages:
            return
        python_path = sys.executable.replace("pythonw.exe", "python.exe")
        self.set_busy(True)
        self._current_action = "deps"
        self.run_process([python_path, "-m", "pip", "install", *packages])

    def generate_audio(self):
        if not self.backend_dir.exists():
            messagebox.showwarning(self.tr("status_error"), self.tr("neutts_msg_not_installed"))
            return
        text = self.input_text.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning(self.tr("status_error"), self.tr("neutts_msg_missing_text"))
            return
        ref_audio = self.ref_audio_var.get().strip()
        if not ref_audio:
            messagebox.showwarning(self.tr("status_error"), self.tr("neutts_msg_missing_ref_audio"))
            return
        ref_text = self.ref_text.get("1.0", "end").strip()
        if not ref_text:
            messagebox.showwarning(self.tr("status_error"), self.tr("neutts_msg_missing_ref_text"))
            return

        script_path = self.data_dir / "neutts_run.py"
        if not script_path.exists():
            messagebox.showwarning(self.tr("status_error"), self.tr("neutts_msg_script_missing"))
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time() * 1000)
        output_path = self.output_dir / f"neutts_{stamp}.wav"
        self._last_output = output_path
        self.output_path_label.configure(text=str(output_path))

        python_path = sys.executable.replace("pythonw.exe", "python.exe")
        device = self.device_var.get().strip() or "cpu"
        cmd = [
            python_path,
            "-u",
            str(script_path),
            "--text",
            text,
            "--ref-audio",
            ref_audio,
            "--ref-text",
            ref_text,
            "--backbone",
            self.current_repo,
            "--codec",
            "neuphonic/neucodec",
            "--device",
            device,
            "--output",
            str(output_path),
        ]
        self.set_busy(True)
        self._current_action = "generate"
        self.run_process(cmd)

    def preview_audio(self):
        if not self._last_output or not self._last_output.exists():
            messagebox.showwarning(self.tr("status_error"), self.tr("neutts_msg_output_missing"))
            return
        try:
            import winsound
            winsound.PlaySound(str(self._last_output), winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as exc:
            messagebox.showwarning(self.tr("status_error"), f"{self.tr('status_error')}: {exc}")

    def stop_preview(self):
        try:
            import winsound
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass

    def save_output(self):
        if not self._last_output or not self._last_output.exists():
            messagebox.showwarning(self.tr("status_error"), self.tr("neutts_msg_output_missing"))
            return
        dest = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV", "*.wav")],
        )
        if not dest:
            return
        try:
            import shutil
            shutil.copy2(self._last_output, dest)
            messagebox.showinfo(self.tr("neutts_btn_save"), self.tr("neutts_msg_saved").format(dest))
        except Exception as exc:
            messagebox.showwarning(self.tr("status_error"), f"{self.tr('status_error')}: {exc}")

    def open_output_folder(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        os.startfile(str(self.output_dir))
