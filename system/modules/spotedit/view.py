import customtkinter as ctk
import os
import sys
import threading
import subprocess
import webbrowser
import time
import tkinter as tk
import logging
from pathlib import Path
from tkinter import messagebox, filedialog

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
        self._busy = False
        self._current_action = None
        self.output_dir = None
        self.base_image = None
        self.mask_image = None
        self.tk_image = None
        self.canvas_image_id = None
        self.display_scale = 1.0
        self.display_size = (0, 0)
        self.display_offset = (0, 0)
        self._pending_retry = None
        self._deps_installed = False
        self._model_options = {
            self.tr("spotedit_model_flux"): "flux",
            self.tr("spotedit_model_qwen"): "qwen",
        }
        self.model_var = ctk.StringVar(value=self.tr("spotedit_model_qwen"))

        # parents[3] points to the app root (UNLZ-AI-STUDIO)
        self.app_root = Path(__file__).resolve().parents[3]
        self.backend_dir = self.app_root / "system" / "ai-backends" / "SpotEdit"
        self.output_dir = self.app_root / "system" / "spotedit-out"

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

        status_frame = ctk.CTkFrame(self, fg_color="transparent")
        status_frame.pack(fill="x", padx=10, pady=(5, 0))
        ctk.CTkLabel(status_frame, text=self.tr("spotedit_status_label")).pack(side="left")
        self.status_value = ctk.CTkLabel(status_frame, text=self.tr("spotedit_status_idle"), text_color="gray")
        self.status_value.pack(side="left", padx=(6, 0))

        editor = ctk.CTkFrame(self, fg_color="transparent")
        editor.pack(fill="both", expand=True, padx=10, pady=(5, 5))
        editor.grid_columnconfigure(0, weight=0)
        editor.grid_columnconfigure(1, weight=1)
        editor.grid_rowconfigure(0, weight=1)

        controls = ctk.CTkFrame(editor)
        controls.grid(row=0, column=0, sticky="nsw", padx=(0, 10), pady=5)

        ctk.CTkLabel(controls, text=self.tr("spotedit_editor_title"), font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=10, pady=(10, 5))
        ctk.CTkLabel(controls, text=self.tr("spotedit_editor_note"), text_color="gray").pack(anchor="w", padx=10, pady=(0, 10))

        ctk.CTkButton(controls, text=self.tr("spotedit_btn_load_image"), command=self.load_image).pack(fill="x", padx=10, pady=4)
        ctk.CTkButton(controls, text=self.tr("spotedit_btn_clear_mask"), command=self.clear_mask).pack(fill="x", padx=10, pady=4)

        ctk.CTkLabel(controls, text=self.tr("spotedit_brush_label")).pack(anchor="w", padx=10, pady=(10, 0))
        self.brush_size = ctk.CTkSlider(controls, from_=5, to=80, number_of_steps=15)
        self.brush_size.set(25)
        self.brush_size.pack(fill="x", padx=10, pady=(0, 10))

        ctk.CTkLabel(controls, text=self.tr("spotedit_model_label")).pack(anchor="w", padx=10, pady=(10, 0))
        self.model_menu = ctk.CTkOptionMenu(controls, variable=self.model_var, values=list(self._model_options.keys()))
        self.model_menu.pack(fill="x", padx=10, pady=(0, 10))

        ctk.CTkButton(controls, text=self.tr("spotedit_btn_download_model"), command=self.download_model).pack(fill="x", padx=10, pady=4)

        ctk.CTkLabel(controls, text=self.tr("spotedit_prompt_label")).pack(anchor="w", padx=10, pady=(10, 0))
        self.prompt_text = ctk.CTkTextbox(
            controls,
            height=80,
            fg_color=("gray95", "#1f1f1f"),
            border_width=1,
            border_color=("gray70", "#2a2a2a"),
        )
        self.prompt_text.pack(fill="x", padx=10, pady=(0, 10))

        ctk.CTkButton(controls, text=self.tr("spotedit_btn_modify"), command=self.run_inpaint).pack(fill="x", padx=10, pady=4)
        ctk.CTkButton(controls, text=self.tr("spotedit_btn_open_output"), command=self.open_output_folder).pack(fill="x", padx=10, pady=4)

        canvas_frame = ctk.CTkFrame(editor)
        canvas_frame.grid(row=0, column=1, sticky="nsew", pady=5)
        canvas_frame.grid_rowconfigure(1, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(canvas_frame, text=self.tr("spotedit_canvas_label")).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 0))
        self.canvas = tk.Canvas(canvas_frame, width=720, height=480, bg="#1b1b1b", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.canvas.bind("<ButtonPress-1>", self.on_paint_start)
        self.canvas.bind("<B1-Motion>", self.on_paint_move)

    def refresh_buttons(self):
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
            deps_marker = self.backend_dir / ".deps_installed"
            if deps_marker.exists():
                self._deps_installed = True
                self.btn_deps.configure(text=self.tr("spotedit_btn_deps_installed"))
            else:
                self._deps_installed = False
                self.btn_deps.configure(text=self.tr("spotedit_btn_deps"))
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
                text=self.tr("status_in_progress") if busy else self.tr("spotedit_status_idle")
            )
        self.refresh_buttons()

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
        self.set_busy(True)
        self._current_action = "install"
        self.run_process(
            ["git", "clone", "--depth", "1", "https://github.com/Biangbiang0321/SpotEdit", str(self.backend_dir)],
            on_done=self.on_process_done,
        )

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
        if self._deps_installed:
            if not messagebox.askyesno(self.tr("spotedit_reinstall_title"), self.tr("spotedit_reinstall_msg")):
                return
        req_path = self.backend_dir / "requirements.txt"
        if not req_path.exists():
            messagebox.showwarning(self.tr("status_error"), self.tr("spotedit_msg_requirements_missing"))
            return
        python_path = sys.executable.replace("pythonw.exe", "python.exe")
        self.log(self.tr("spotedit_msg_installing_deps"))
        self.set_busy(True)
        self._current_action = "deps"
        self.run_process([python_path, "-m", "pip", "install", "-r", str(req_path)], on_done=self.on_process_done)

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
                if process.returncode == 0:
                    self.safe_log(self.tr("spotedit_msg_done"))
                else:
                    self.safe_log(self.tr("spotedit_msg_failed").format(process.returncode))
            except Exception as exc:
                returncode = 1
                self.safe_log(f"{self.tr('status_error')}: {exc}")
            if on_done:
                self.after(0, lambda: on_done(returncode))
            elif self._busy:
                self.after(0, lambda: self.on_process_done(returncode))

        threading.Thread(target=worker, daemon=True).start()

    def log(self, message):
        logging.info(message)

    def safe_log(self, message):
        self.after(0, lambda: logging.info(message))

    def on_process_done(self, returncode):
        self.set_busy(False)
        if returncode == 0 and self._current_action == "deps":
            self.log(self.tr("spotedit_msg_deps_done"))
            try:
                (self.backend_dir / ".deps_installed").write_text("ok", encoding="utf-8")
            except Exception:
                pass
        self._current_action = None
        self.refresh_buttons()

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title=self.tr("spotedit_dialog_load"),
            filetypes=[
                ("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.webp"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return
        try:
            from PIL import Image
        except Exception:
            self.install_python_packages(["pillow"], self.load_image)
            return
        try:
            image = Image.open(file_path).convert("RGB")
        except Exception as exc:
            messagebox.showwarning(self.tr("status_error"), f"{self.tr('status_error')}: {exc}")
            return
        self.base_image = image
        self.mask_image = Image.new("L", image.size, 0)
        self.render_canvas()
        self.log(self.tr("spotedit_msg_loaded").format(file_path))

    def clear_mask(self):
        if not self.base_image:
            messagebox.showwarning(self.tr("status_error"), self.tr("spotedit_msg_no_image"))
            return
        try:
            from PIL import Image
        except Exception:
            self.install_python_packages(["pillow"], self.clear_mask)
            return
        self.mask_image = Image.new("L", self.base_image.size, 0)
        self.render_canvas()
        self.log(self.tr("spotedit_msg_mask_cleared"))

    def on_paint_start(self, event):
        self.paint_at(event.x, event.y)

    def on_paint_move(self, event):
        self.paint_at(event.x, event.y)

    def paint_at(self, x, y):
        if not self.base_image or not self.mask_image:
            return
        if not self.display_size[0] or not self.display_size[1]:
            return
        img_coords = self.canvas_to_image(x, y)
        if not img_coords:
            return
        try:
            from PIL import ImageDraw
        except Exception:
            self.install_python_packages(["pillow"], None)
            return
        radius = int(self.brush_size.get())
        radius = max(2, radius)
        img_radius = max(1, int(radius / max(self.display_scale, 0.01)))
        draw = ImageDraw.Draw(self.mask_image)
        cx, cy = img_coords
        draw.ellipse((cx - img_radius, cy - img_radius, cx + img_radius, cy + img_radius), fill=255)
        self.render_canvas()

    def canvas_to_image(self, x, y):
        offset_x, offset_y = self.display_offset
        disp_w, disp_h = self.display_size
        if x < offset_x or y < offset_y:
            return None
        rel_x = x - offset_x
        rel_y = y - offset_y
        if rel_x > disp_w or rel_y > disp_h:
            return None
        img_x = int(rel_x / max(self.display_scale, 0.01))
        img_y = int(rel_y / max(self.display_scale, 0.01))
        return img_x, img_y

    def render_canvas(self):
        if not self.base_image:
            self.canvas.delete("all")
            self.tk_image = None
            self.canvas_image_id = None
            return
        try:
            from PIL import Image, ImageTk
        except Exception:
            self.install_python_packages(["pillow"], self.render_canvas)
            return

        base = self.base_image.convert("RGBA")
        if self.mask_image:
            alpha = self.mask_image.point(lambda v: 120 if v > 0 else 0)
            overlay = Image.new("RGBA", base.size, (255, 0, 0, 0))
            overlay.putalpha(alpha)
            base = Image.alpha_composite(base, overlay)

        canvas_w = int(self.canvas.winfo_width() or 720)
        canvas_h = int(self.canvas.winfo_height() or 480)
        scale = min(canvas_w / base.width, canvas_h / base.height)
        new_w = max(1, int(base.width * scale))
        new_h = max(1, int(base.height * scale))
        display = base.resize((new_w, new_h), Image.LANCZOS)
        offset_x = int((canvas_w - new_w) / 2)
        offset_y = int((canvas_h - new_h) / 2)

        self.display_scale = scale
        self.display_size = (new_w, new_h)
        self.display_offset = (offset_x, offset_y)

        canvas_img = Image.new("RGBA", (canvas_w, canvas_h), (27, 27, 27, 255))
        canvas_img.paste(display, (offset_x, offset_y), display)
        self.tk_image = ImageTk.PhotoImage(canvas_img)
        if self.canvas_image_id:
            self.canvas.itemconfig(self.canvas_image_id, image=self.tk_image)
        else:
            self.canvas_image_id = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

    def run_inpaint(self):
        if not self.base_image:
            messagebox.showwarning(self.tr("status_error"), self.tr("spotedit_msg_no_image"))
            return
        if not self.mask_image or self.mask_image.getbbox() is None:
            messagebox.showwarning(self.tr("status_error"), self.tr("spotedit_msg_no_mask"))
            return
        if not self.backend_dir.exists():
            messagebox.showwarning(self.tr("status_error"), self.tr("spotedit_msg_not_installed"))
            return
        self.log(self.tr("spotedit_msg_inpaint_start"))
        self.log(self.tr("spotedit_msg_first_run"))
        prompt = self.prompt_text.get("1.0", "end").strip()
        if prompt:
            self.log(self.tr("spotedit_msg_prompt").format(prompt))
        try:
            from PIL import Image
        except Exception:
            self.install_python_packages(["pillow"], self.run_inpaint)
            return

        temp_dir = self.app_root / "system" / "data" / "spotedit"
        temp_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time() * 1000)
        input_path = temp_dir / f"spotedit_input_{stamp}.png"
        mask_path = temp_dir / f"spotedit_mask_{stamp}.png"
        output_path = (self.output_dir / f"spotedit_{stamp}.png")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.base_image.convert("RGB").save(input_path)
        self.mask_image.convert("L").save(mask_path)

        backend_key = self._model_options.get(self.model_var.get(), "qwen")
        script_path = self.app_root / "system" / "data" / "spotedit" / "spotedit_run.py"
        python_path = sys.executable.replace("pythonw.exe", "python.exe")
        self.set_busy(True)
        self._current_action = "run"
        self.run_process(
            [
                python_path,
                "-u",
                str(script_path),
                "--backend",
                backend_key,
                "--input",
                str(input_path),
                "--mask",
                str(mask_path),
                "--output",
                str(output_path),
                "--prompt",
                prompt or self.tr("spotedit_prompt_default"),
            ],
            on_done=lambda code: self.on_inpaint_done(code, output_path),
        )

    def download_model(self):
        if not self.backend_dir.exists():
            messagebox.showwarning(self.tr("status_error"), self.tr("spotedit_msg_not_installed"))
            return
        backend_key = self._model_options.get(self.model_var.get(), "qwen")
        script_path = self.app_root / "system" / "data" / "spotedit" / "spotedit_run.py"
        python_path = sys.executable.replace("pythonw.exe", "python.exe")
        self.log(self.tr("spotedit_msg_downloading_model").format(self.model_var.get()))
        self.set_busy(True)
        self._current_action = "download"
        self.run_process(
            [
                python_path,
                "-u",
                str(script_path),
                "--backend",
                backend_key,
                "--input",
                str(script_path),
                "--mask",
                str(script_path),
                "--output",
                str(script_path),
                "--prompt",
                self.tr("spotedit_prompt_default"),
                "--download-only",
            ],
            on_done=self.on_download_done,
        )

    def on_download_done(self, returncode):
        self.set_busy(False)
        if returncode == 0:
            self.log(self.tr("spotedit_msg_download_done"))
        else:
            self.log(self.tr("spotedit_msg_failed").format(returncode))
        self._current_action = None
        self.refresh_buttons()

    def on_inpaint_done(self, returncode, output_path):
        self.set_busy(False)
        if returncode != 0:
            self.log(self.tr("spotedit_msg_failed").format(returncode))
            return
        try:
            from PIL import Image
        except Exception:
            self.log(self.tr("spotedit_msg_saved").format(output_path))
            return
        if output_path.exists():
            try:
                self.base_image = Image.open(output_path).convert("RGB")
                self.mask_image = Image.new("L", self.base_image.size, 0)
                self.render_canvas()
            except Exception:
                pass
        self.log(self.tr("spotedit_msg_saved").format(output_path))
        out_dir = self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"spotedit_{int(time.time() * 1000)}.png"
        Image.fromarray(output).save(out_path)
        self.base_image = Image.fromarray(output).convert("RGB")
        self.mask_image = Image.new("L", self.base_image.size, 0)
        self.render_canvas()
        self.log(self.tr("spotedit_msg_saved").format(out_path))

    def open_output_folder(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        os.startfile(str(self.output_dir))

    def install_python_packages(self, packages, retry_callback):
        if not packages:
            return
        if retry_callback:
            self._pending_retry = retry_callback
        python_path = sys.executable.replace("pythonw.exe", "python.exe")
        self.log(self.tr("spotedit_msg_installing_packages").format(", ".join(packages)))
        self.run_process([python_path, "-m", "pip", "install", *packages], on_done=self.on_packages_done)

    def on_packages_done(self, returncode):
        if returncode != 0:
            self.log(self.tr("spotedit_msg_packages_failed"))
            self._pending_retry = None
            return
        self.log(self.tr("spotedit_msg_packages_done"))
        if self._pending_retry:
            callback = self._pending_retry
            self._pending_retry = None
            self.after(0, callback)
