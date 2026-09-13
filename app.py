"""Frame Annotator: pick frames from an .avi, run person detection, save annotated JPGs.

Run with the vision conda env:
    ~/miniconda3/envs/vision/bin/python ~/projekts/sandbox/frame_annotator/app.py
"""
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

import annotate as ann

RECORDINGS_DIR = Path("~/projekts/httprequests/recordings").expanduser()
DEFAULT_OUT = Path("~/projekts/sandbox/annotated").expanduser()
PREVIEW_W, PREVIEW_H = 960, 540
CROP_COLORS = {(800, 600): "#ffb000", (640, 480): "#00c8ff", (320, 240): "#ff4fd8"}


class App:
    def __init__(self, root):
        self.root = root
        self.q = queue.Queue()

        self.video_path = None
        self.cap = None
        self.frame_count = 0
        self.fps = 30.0
        self.full_size = None
        self.idx = 0
        self.frame = None  # raw BGR frame at self.idx
        self.marks = {}  # video path -> set of frame indices

        self.model = None
        self.model_name = ann.DEFAULT_MODEL
        self.model_loading = False
        self.model_lock = threading.Lock()  # preview and save threads share the model
        self.det_running = False
        self.busy = False
        self.out_dir = DEFAULT_OUT
        self.photo = None

        self._build_ui()
        self._bind_keys()
        self._load_model(self.model_name)
        self.root.after(50, self._poll)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- UI construction ----------

    def _build_ui(self):
        self.root.title("Frame Annotator")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        top = ttk.Frame(self.root, padding=(8, 6))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)
        top.columnconfigure(5, weight=1)
        ttk.Button(top, text="Open video…", command=self.open_video, takefocus=False).grid(row=0, column=0)
        self.video_label = ttk.Label(top, text="No video loaded", width=40)
        self.video_label.grid(row=0, column=1, sticky="w", padx=(6, 12))
        ttk.Label(top, text="Model:").grid(row=0, column=2)
        self.model_var = tk.StringVar(value=self.model_name)
        self.model_combo = ttk.Combobox(top, textvariable=self.model_var, state="readonly",
                                        values=ann.available_models(), width=22)
        self.model_combo.grid(row=0, column=3, padx=(4, 12))
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_selected)
        ttk.Button(top, text="Output folder…", command=self.choose_out_dir, takefocus=False).grid(row=0, column=4)
        self.out_label = ttk.Label(top, text=str(self.out_dir))
        self.out_label.grid(row=0, column=5, sticky="w", padx=6)

        main = ttk.Frame(self.root, padding=(8, 0))
        main.grid(row=1, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        # left: preview + navigation
        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="nsew")
        self.canvas = tk.Canvas(left, width=PREVIEW_W, height=PREVIEW_H, bg="black", highlightthickness=0)
        self.canvas.grid(row=0, column=0, columnspan=2)
        self.canvas.bind("<Button-1>", lambda e: self.root.focus_set())

        self.slider = tk.Scale(left, orient="horizontal", from_=0, to=0, showvalue=False,
                               command=self._on_slider, takefocus=0)
        self.slider.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        nav = ttk.Frame(left)
        nav.grid(row=2, column=0, sticky="w", pady=6)
        for text, step in (("−10", -10), ("−1", -1), ("+1", 1), ("+10", 10)):
            ttk.Button(nav, text=text, width=4, takefocus=False,
                       command=lambda s=step: self.step(s)).pack(side="left", padx=2)
        self.mark_btn = ttk.Button(nav, text="Mark (Space)", takefocus=False, command=self.toggle_mark)
        self.mark_btn.pack(side="left", padx=(12, 2))
        self.frame_label = tk.Label(left, text="", font=("TkDefaultFont", 11))
        self.frame_label.grid(row=2, column=1, sticky="e")

        # right: marks, sizes, save
        right = ttk.Frame(main, padding=(12, 0, 0, 0))
        right.grid(row=0, column=1, sticky="ns")
        right.rowconfigure(1, weight=1)

        ttk.Label(right, text="Marked frames").grid(row=0, column=0, sticky="w")
        lb_frame = ttk.Frame(right)
        lb_frame.grid(row=1, column=0, sticky="nsew")
        self.listbox = tk.Listbox(lb_frame, width=22, height=12, exportselection=False)
        self.listbox.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(lb_frame, orient="vertical", command=self.listbox.yview)
        sb.pack(side="right", fill="y")
        self.listbox.config(yscrollcommand=sb.set)
        self.listbox.bind("<Double-Button-1>", self._on_list_jump)
        self.listbox.bind("<Delete>", self._on_list_delete)
        self.listbox.bind("<BackSpace>", self._on_list_delete)
        ttk.Label(right, text="Double-click: jump · Delete: remove", foreground="gray").grid(
            row=2, column=0, sticky="w")
        ttk.Button(right, text="Clear marks", command=self.clear_marks, takefocus=False).grid(
            row=3, column=0, sticky="ew", pady=(4, 12))

        sizes = ttk.LabelFrame(right, text="Output sizes (centred crop)", padding=6)
        sizes.grid(row=4, column=0, sticky="ew")
        self.size_vars = {}
        for i, size in enumerate(ann.CROP_SIZES):
            var = tk.BooleanVar(value=(i == 0))
            var.trace_add("write", lambda *_: self._on_sizes_changed())
            self.size_vars[size] = var
            label = f"{size[0]}×{size[1]}" + (" (full frame)" if i == 0 else "")
            row = ttk.Frame(sizes)
            row.pack(anchor="w")
            ttk.Checkbutton(row, text=label, variable=var, takefocus=False).pack(side="left")
            if size in CROP_COLORS:
                tk.Label(row, text="■", fg=CROP_COLORS[size]).pack(side="left")

        self.show_det = tk.BooleanVar(value=False)
        self.show_det_cb = ttk.Checkbutton(right, text="Show detections in preview", variable=self.show_det,
                                           command=self.render, takefocus=False)
        self.show_det_cb.grid(row=5, column=0, sticky="w", pady=(10, 4))

        self.save_btn = ttk.Button(right, text="Annotate & save marked", command=self.save_marked,
                                   takefocus=False)
        self.save_btn.grid(row=6, column=0, sticky="ew", pady=(6, 4))
        self.progress = ttk.Progressbar(right, mode="determinate")
        self.progress.grid(row=7, column=0, sticky="ew")

        self.status = ttk.Label(self.root, text="", anchor="w", padding=(8, 4), relief="sunken")
        self.status.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        self._update_controls()

    def _bind_keys(self):
        self.root.bind("<Left>", lambda e: self._key_step(e, -1))
        self.root.bind("<Right>", lambda e: self._key_step(e, 1))
        self.root.bind("<Shift-Left>", lambda e: self._key_step(e, -10))
        self.root.bind("<Shift-Right>", lambda e: self._key_step(e, 10))
        self.root.bind("<space>", lambda e: self.toggle_mark())

    def _key_step(self, event, n):
        if isinstance(event.widget, tk.Scale):
            return  # the slider handles its own arrow keys
        self.step(n)

    # ---------- model ----------

    def _load_model(self, name):
        self.model_loading = True
        self.model = None
        self._set_status(f"Loading model {name}…")
        self._update_controls()

        def work():
            try:
                self.q.put(("model_ready", name, ann.load_model(name)))
            except Exception as e:
                self.q.put(("model_error", name, str(e)))

        threading.Thread(target=work, daemon=True).start()

    def _on_model_selected(self, _event):
        self.root.focus_set()
        name = self.model_var.get()
        if name != self.model_name or self.model is None:
            self.model_name = name
            self._load_model(name)

    # ---------- video and navigation ----------

    def open_video(self):
        initial = self.video_path.parent if self.video_path else RECORDINGS_DIR
        path = filedialog.askopenfilename(title="Open video", initialdir=initial,
                                          filetypes=[("AVI video", "*.avi"), ("All files", "*.*")])
        if not path:
            return
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        first = ann.read_frame(cap, 0) if n > 0 else None
        if first is None:
            cap.release()
            messagebox.showerror("Open video", f"Could not read frames from:\n{path}")
            return
        if self.cap is not None:
            self.cap.release()
        self.cap = cap
        self.video_path = Path(path)
        self.frame_count = n
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.full_size = (first.shape[1], first.shape[0])
        self.marks.setdefault(str(self.video_path), set())
        self.video_label.config(text=f"{self.video_path.name}  ({self.full_size[0]}×{self.full_size[1]}, "
                                     f"{n} frames @ {self.fps:.0f} fps)")
        self.slider.config(to=n - 1)
        self._refresh_list()
        self._set_status(f"Opened {self.video_path}")
        self.goto(0, force=True)

    def choose_out_dir(self):
        path = filedialog.askdirectory(title="Output folder", initialdir=self.out_dir, mustexist=False)
        if path:
            self.out_dir = Path(path)
            self.out_label.config(text=str(self.out_dir))

    def goto(self, idx, force=False):
        if self.cap is None:
            return
        idx = max(0, min(self.frame_count - 1, idx))
        if idx == self.idx and not force:
            return
        self.idx = idx
        self.frame = ann.read_frame(self.cap, idx)
        if int(float(self.slider.get())) != idx:
            self.slider.set(idx)
        self.render()

    def step(self, n):
        self.goto(self.idx + n)

    def _on_slider(self, value):
        self.goto(int(float(value)))

    # ---------- marks ----------

    def _current_marks(self):
        return self.marks.get(str(self.video_path), set()) if self.video_path else set()

    def toggle_mark(self):
        if self.video_path is None:
            return
        marks = self._current_marks()
        marks.symmetric_difference_update({self.idx})
        self._refresh_list()
        self._update_frame_label()
        self._update_controls()

    def clear_marks(self):
        if self.video_path is not None:
            self._current_marks().clear()
            self._refresh_list()
            self._update_frame_label()
            self._update_controls()

    def _refresh_list(self):
        self.listbox.delete(0, "end")
        self.list_items = sorted(self._current_marks())
        for i in self.list_items:
            self.listbox.insert("end", f"{i:6d}   ({i / self.fps:6.2f} s)")

    def _on_list_jump(self, _event):
        sel = self.listbox.curselection()
        if sel:
            self.goto(self.list_items[sel[0]])

    def _on_list_delete(self, _event):
        sel = self.listbox.curselection()
        if sel:
            self._current_marks().discard(self.list_items[sel[0]])
            self._refresh_list()
            if self.list_items:
                self.listbox.selection_set(min(sel[0], len(self.list_items) - 1))
            self._update_frame_label()
            self._update_controls()
        return "break"

    # ---------- rendering ----------

    def render(self):
        if self.cap is None:
            return
        self._update_frame_label()
        if self.frame is None:
            self.canvas.delete("all")
            self.canvas.create_text(PREVIEW_W // 2, PREVIEW_H // 2, fill="white",
                                    text=f"Frame {self.idx} could not be read")
            return
        self._show(self.frame)
        if self.show_det.get() and self.model is not None:
            self._request_detection()

    def _show(self, bgr):
        H, W = bgr.shape[:2]
        s = min(PREVIEW_W / W, PREVIEW_H / H)
        dw, dh = int(W * s), int(H * s)
        ox, oy = (PREVIEW_W - dw) // 2, (PREVIEW_H - dh) // 2
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).resize((dw, dh), Image.BILINEAR)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(ox, oy, anchor="nw", image=self.photo)

        for size, color in CROP_COLORS.items():
            if not self.size_vars[size].get() or size[0] > W or size[1] > H:
                continue
            x0, y0 = (W - size[0]) // 2, (H - size[1]) // 2
            self.canvas.create_rectangle(ox + x0 * s, oy + y0 * s, ox + (x0 + size[0]) * s,
                                         oy + (y0 + size[1]) * s, outline=color, width=2, dash=(6, 4))
            self.canvas.create_text(ox + x0 * s + 4, oy + y0 * s + 4, anchor="nw", fill=color,
                                    text=f"{size[0]}×{size[1]}", font=("TkDefaultFont", 10, "bold"))

    def _update_frame_label(self):
        if self.cap is None:
            self.frame_label.config(text="")
            return
        marked = self.idx in self._current_marks()
        text = f"{self.idx} / {self.frame_count - 1}   ({self.idx / self.fps:.2f} s)"
        self.frame_label.config(text=text + ("   ● MARKED" if marked else ""),
                                fg="#d00000" if marked else "black")
        self.mark_btn.config(text="Unmark (Space)" if marked else "Mark (Space)")

    def _request_detection(self):
        # one detection at a time; when it finishes, re-run if the frame changed meanwhile
        if self.det_running:
            return
        self.det_running = True
        video, idx, frame, model = self.video_path, self.idx, self.frame, self.model

        def work():
            try:
                with self.model_lock:
                    img = ann.annotate(model, frame)
                self.q.put(("preview", video, idx, img))
            except Exception as e:
                self.q.put(("preview_error", str(e)))

        threading.Thread(target=work, daemon=True).start()

    def _on_sizes_changed(self):
        self._update_controls()
        self.render()

    # ---------- saving ----------

    def save_marked(self):
        marks = sorted(self._current_marks())
        sizes = [s for s in ann.CROP_SIZES if self.size_vars[s].get()]
        if not (self.video_path and marks and sizes and self.model):
            return
        self.busy = True
        self.progress.config(maximum=len(marks) * len(sizes), value=0)
        self._set_status(f"Annotating {len(marks)} frame(s) × {len(sizes)} size(s)…")
        self._update_controls()
        args = (self.video_path, marks, sizes, self.out_dir, self.model)
        threading.Thread(target=self._save_worker, args=args, daemon=True).start()

    def _save_worker(self, video, marks, sizes, out_dir, model):
        cap = cv2.VideoCapture(str(video))
        done = saved = 0
        skipped, errors = set(), []
        try:
            for idx in marks:
                frame = ann.read_frame(cap, idx)
                if frame is None:
                    errors.append(f"frame {idx}: could not be read")
                    done += len(sizes)
                    self.q.put(("progress", done))
                    continue
                full = (frame.shape[1], frame.shape[0])
                for size in sizes:
                    try:
                        crop = ann.center_crop(frame, *size)
                        with self.model_lock:
                            img = ann.annotate(model, crop, idx)
                        ann.save_jpg(ann.output_path(out_dir, video, idx, size, full), img)
                        saved += 1
                    except ValueError:
                        skipped.add(size)
                    except Exception as e:
                        errors.append(f"frame {idx} {size[0]}x{size[1]}: {e}")
                    done += 1
                    self.q.put(("progress", done))
        finally:
            cap.release()
            self.q.put(("save_done", saved, sorted(skipped), errors, Path(out_dir) / video.stem))

    # ---------- event queue and state ----------

    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "model_ready":
                    _, name, model = msg
                    if name == self.model_name:
                        self.model, self.model_loading = model, False
                        self._set_status(f"Model {name} ready")
                        self._update_controls()
                        self.render()
                elif kind == "model_error":
                    _, name, err = msg
                    if name == self.model_name:
                        self.model_loading = False
                        self._set_status(f"Failed to load model {name}: {err}")
                        self._update_controls()
                elif kind == "preview":
                    _, video, idx, img = msg
                    self.det_running = False
                    if (video, idx) == (self.video_path, self.idx) and self.show_det.get():
                        self._show(img)
                    elif self.show_det.get():
                        self.render()
                elif kind == "preview_error":
                    self.det_running = False
                    self._set_status(f"Detection failed: {msg[1]}")
                elif kind == "progress":
                    self.progress.config(value=msg[1])
                elif kind == "save_done":
                    _, saved, skipped, errors, folder = msg
                    self.busy = False
                    text = f"Saved {saved} image(s) to {folder}"
                    if skipped:
                        text += "  · skipped (larger than frame): " + ", ".join(f"{w}x{h}" for w, h in skipped)
                    if errors:
                        text += f"  · {len(errors)} error(s)"
                        messagebox.showwarning("Save", "\n".join(errors[:20]))
                    self._set_status(text)
                    self._update_controls()
        except queue.Empty:
            pass
        self.root.after(50, self._poll)

    def _update_controls(self):
        ready = self.model is not None and not self.model_loading
        can_save = (ready and not self.busy and self.video_path is not None
                    and bool(self._current_marks()) and any(v.get() for v in self.size_vars.values()))
        self.save_btn.state(["!disabled"] if can_save else ["disabled"])
        self.show_det_cb.state(["!disabled"] if ready else ["disabled"])
        self.model_combo.state(["disabled"] if self.busy or self.model_loading else ["!disabled", "readonly"])

    def _set_status(self, text):
        self.status.config(text=text)

    def _on_close(self):
        if self.busy and not messagebox.askyesno("Quit", "Saving is still in progress. Quit anyway?"):
            return
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
