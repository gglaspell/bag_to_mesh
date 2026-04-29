#!/usr/bin/env python3
"""
bag_to_mesh_gui.py — Tkinter GUI for the bag-to-mesh Docker container.
Builds and runs the docker run command with all supported parameters.
"""
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import subprocess
import threading
import shlex
from pathlib import Path

# ── colour tokens ─────────────────────────────────────────────────────────────
BG        = "#1e1e2e"
SURFACE   = "#252538"
SURFACE2  = "#2e2e45"
SURFACE3  = "#383855"
ACCENT    = "#7c9ef5"
ACCENT_DK = "#5a7cd4"
TEXT      = "#cdd6f4"
MUTED     = "#a6adc8"
FAINT     = "#585b70"
BORDER    = "#45475a"
ENTRY_BG  = "#1c1c2e"
SUCCESS   = "#a6e3a1"
ERROR     = "#f38ba8"
WARN      = "#fab387"

FONT_BODY  = ("Segoe UI", 10)
FONT_SMALL = ("Segoe UI", 9)
FONT_MONO  = ("Consolas", 9)
FONT_HEAD  = ("Segoe UI Semibold", 11)
FONT_TITLE = ("Segoe UI", 15, "bold")
FONT_SEMI  = ("Segoe UI Semibold", 10)


# ── reusable widgets ──────────────────────────────────────────────────────────

class PathEntry(tk.Frame):
    """Label + entry + browse button, themed."""
    def __init__(self, parent, label, default="", tip="", browse_fn=None, **kw):
        super().__init__(parent, bg=SURFACE, **kw)
        lrow = tk.Frame(self, bg=SURFACE)
        lrow.pack(fill="x")
        tk.Label(lrow, text=label, bg=SURFACE, fg=TEXT,
                 font=FONT_SEMI, anchor="w").pack(side="left")
        if tip:
            tk.Label(lrow, text=tip, bg=SURFACE, fg=FAINT,
                     font=("Segoe UI", 8), anchor="w").pack(side="left", padx=4)
        row = tk.Frame(self, bg=SURFACE)
        row.pack(fill="x", pady=(2, 0))
        self.var = tk.StringVar(value=default)
        entry = tk.Entry(row, textvariable=self.var, bg=ENTRY_BG, fg=TEXT,
                         insertbackground=TEXT, relief="flat", font=FONT_BODY,
                         highlightthickness=1, highlightbackground=BORDER,
                         highlightcolor=ACCENT)
        entry.pack(side="left", fill="x", expand=True, ipady=5, padx=(0, 6))
        if browse_fn:
            tk.Button(row, text="Browse...", command=browse_fn,
                      bg=SURFACE3, fg=TEXT, relief="flat", font=FONT_SMALL,
                      padx=10, pady=4, cursor="hand2",
                      activebackground=BORDER, activeforeground=TEXT
                      ).pack(side="left")

    def get(self):    return self.var.get().strip()
    def set(self, v): self.var.set(v)


class TextEntry(tk.Frame):
    """Label + plain entry, themed."""
    def __init__(self, parent, label, default="", tip="", **kw):
        super().__init__(parent, bg=SURFACE, **kw)
        lrow = tk.Frame(self, bg=SURFACE)
        lrow.pack(fill="x")
        tk.Label(lrow, text=label, bg=SURFACE, fg=TEXT,
                 font=FONT_SEMI, anchor="w").pack(side="left")
        if tip:
            tk.Label(lrow, text=tip, bg=SURFACE, fg=FAINT,
                     font=("Segoe UI", 8), anchor="w").pack(side="left", padx=4)
        self.var = tk.StringVar(value=default)
        entry = tk.Entry(self, textvariable=self.var, bg=ENTRY_BG, fg=TEXT,
                         insertbackground=TEXT, relief="flat", font=FONT_BODY,
                         highlightthickness=1, highlightbackground=BORDER,
                         highlightcolor=ACCENT)
        entry.pack(fill="x", ipady=5, pady=(2, 0))

    def get(self):    return self.var.get().strip()
    def set(self, v): self.var.set(v)


class SliderRow(tk.Frame):
    """Label + range hint + Scale + numeric Entry, fully synced."""
    def __init__(self, parent, label, default, from_, to,
                 resolution=0.001, is_int=False, tip="", **kw):
        super().__init__(parent, bg=SURFACE, **kw)
        self.is_int = is_int
        self.var = tk.DoubleVar(value=float(default))
        self._updating = False

        hrow = tk.Frame(self, bg=SURFACE)
        hrow.pack(fill="x")
        tk.Label(hrow, text=label, bg=SURFACE, fg=TEXT,
                 font=FONT_SEMI, anchor="w").pack(side="left")
        tk.Label(hrow, text=f"  [{from_} - {to}]", bg=SURFACE, fg=FAINT,
                 font=("Segoe UI", 8)).pack(side="left")
        if tip:
            tk.Label(hrow, text=f"  {tip}", bg=SURFACE, fg=FAINT,
                     font=("Segoe UI", 8)).pack(side="left")

        row = tk.Frame(self, bg=SURFACE)
        row.pack(fill="x", pady=(2, 0))

        self.scale = tk.Scale(
            row, from_=from_, to=to, orient="horizontal",
            variable=self.var, resolution=resolution,
            bg=SURFACE, fg=TEXT, troughcolor=ENTRY_BG,
            highlightthickness=0, sliderrelief="flat",
            activebackground=ACCENT, showvalue=False,
            command=self._on_scale)
        self.scale.pack(side="left", fill="x", expand=True)

        self.entry = tk.Entry(row, width=9, bg=ENTRY_BG, fg=ACCENT,
                              insertbackground=ACCENT, relief="flat",
                              font=FONT_MONO, justify="center",
                              highlightthickness=1, highlightbackground=BORDER,
                              highlightcolor=ACCENT)
        self.entry.insert(0, self._fmt(default))
        self.entry.pack(side="left", padx=(8, 0), ipady=5)
        self.entry.bind("<FocusOut>", self._on_entry)
        self.entry.bind("<Return>",   self._on_entry)

    def _fmt(self, v):
        return str(int(round(v))) if self.is_int else f"{float(v):.4g}"

    def _on_scale(self, _=None):
        if self._updating:
            return
        self._updating = True
        self.entry.delete(0, "end")
        self.entry.insert(0, self._fmt(self.var.get()))
        self._updating = False

    def _on_entry(self, _=None):
        if self._updating:
            return
        try:
            v = float(self.entry.get())
            self._updating = True
            self.var.set(v)
            self._updating = False
        except ValueError:
            pass

    def get(self):
        try:
            v = float(self.entry.get())
            return int(round(v)) if self.is_int else v
        except ValueError:
            v = self.var.get()
            return int(round(v)) if self.is_int else v


class CheckRow(tk.Frame):
    """Styled checkbox."""
    def __init__(self, parent, label, default=False, tip="", **kw):
        super().__init__(parent, bg=SURFACE, **kw)
        frame = tk.Frame(self, bg=SURFACE)
        frame.pack(anchor="w", fill="x")
        self.var = tk.BooleanVar(value=default)
        cb = tk.Checkbutton(frame, text=label, variable=self.var,
                            bg=SURFACE, fg=TEXT, selectcolor=ENTRY_BG,
                            activebackground=SURFACE, activeforeground=TEXT,
                            font=FONT_SEMI, anchor="w", cursor="hand2")
        cb.pack(side="left")
        if tip:
            tk.Label(frame, text=tip, bg=SURFACE, fg=FAINT,
                     font=("Segoe UI", 8)).pack(side="left", padx=6)

    def get(self): return self.var.get()


class Section(tk.Frame):
    """Themed section with accent header bar."""
    def __init__(self, parent, title, **kw):
        super().__init__(parent, bg=SURFACE, bd=0, **kw)
        tk.Frame(self, bg=ACCENT_DK, height=2).pack(fill="x")
        tk.Label(self, text=title, bg=SURFACE2, fg=ACCENT,
                 font=FONT_HEAD, anchor="w", padx=10, pady=6
                 ).pack(fill="x")
        self.inner = tk.Frame(self, bg=SURFACE, padx=14, pady=8)
        self.inner.pack(fill="both", expand=True)

    def pad(self):
        return dict(fill="x", pady=5)


# ── main application ──────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("bag-to-mesh  -  Docker GUI")
        self.configure(bg=BG)
        self.minsize(900, 640)
        self.geometry("1100x760")

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Vertical.TScrollbar",
                        background=SURFACE2, troughcolor=BG,
                        arrowcolor=MUTED, bordercolor=BG, relief="flat")

        self._build_ui()
        self._update_command()

    # ── layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # title bar
        hdr = tk.Frame(self, bg=BG, pady=10)
        hdr.pack(fill="x", padx=20)
        tk.Label(hdr, text="bag-to-mesh", bg=BG, fg=ACCENT,
                 font=FONT_TITLE).pack(side="left")
        tk.Label(hdr, text="  Docker GUI", bg=BG, fg=MUTED,
                 font=("Segoe UI", 13)).pack(side="left", pady=2)
        tk.Label(hdr, text="ROS 2 Bag -> 3D Mesh Converter", bg=BG, fg=FAINT,
                 font=("Segoe UI", 9)).pack(side="right")

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        # ── bottom action bar (must be packed BEFORE the expanding body)
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", side="bottom")
        bar = tk.Frame(self, bg=SURFACE2, pady=8)
        bar.pack(fill="x", side="bottom")

        self.run_btn = tk.Button(
            bar, text="Run Conversion", command=self._run,
            bg=ACCENT, fg="#1e1e2e", relief="flat",
            font=("Segoe UI Semibold", 11), padx=26, pady=7, cursor="hand2",
            activebackground=ACCENT_DK, activeforeground="#1e1e2e")
        self.run_btn.pack(side="right", padx=16)

        tk.Button(bar, text="Copy Command", command=self._copy_cmd,
                  bg=SURFACE3, fg=TEXT, relief="flat", font=FONT_BODY,
                  padx=14, pady=7, cursor="hand2",
                  activebackground=BORDER, activeforeground=TEXT
                  ).pack(side="right", padx=4)

        tk.Button(bar, text="Clear Log", command=self._clear_log,
                  bg=SURFACE3, fg=MUTED, relief="flat", font=FONT_BODY,
                  padx=12, pady=7, cursor="hand2",
                  activebackground=BORDER, activeforeground=TEXT
                  ).pack(side="left", padx=16)

        # two-pane body
        body = tk.PanedWindow(self, orient="horizontal", bg=BG,
                              sashwidth=6, sashrelief="flat",
                              sashpad=0, opaqueresize=True)
        body.pack(fill="both", expand=True)

        # LEFT — scrollable parameter pane
        left_outer = tk.Frame(body, bg=BG)
        body.add(left_outer, minsize=420, width=500)

        canvas = tk.Canvas(left_outer, bg=BG, highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(left_outer, orient="vertical",
                                  command=canvas.yview,
                                  style="Vertical.TScrollbar")
        self._param_frame = tk.Frame(canvas, bg=BG)
        self._param_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        self._cwin = canvas.create_window((0, 0), window=self._param_frame,
                                          anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(self._cwin, width=e.width))
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _scroll(e):
            delta = -1 * (e.delta // 120) if e.delta else (-1 if e.num == 4 else 1)
            canvas.yview_scroll(delta, "units")
        canvas.bind_all("<MouseWheel>", _scroll)
        canvas.bind_all("<Button-4>",   _scroll)
        canvas.bind_all("<Button-5>",   _scroll)

        self._build_params(self._param_frame)

        # RIGHT — command preview + log
        right = tk.Frame(body, bg=BG)
        body.add(right, minsize=360)
        self._build_right(right)



    # ── parameter widgets ─────────────────────────────────────────────────────

    def _build_params(self, parent):
        p = dict(fill="x", padx=10, pady=5)

        # Required
        s = Section(parent, "Required Inputs")
        s.pack(fill="x", padx=8, pady=(8, 6))

        self.bag_path = PathEntry(
            s.inner, "bag_path", browse_fn=self._browse_bag,
            tip="Path to the ROS 2 .bag / .db3 / .mcap file")
        self.bag_path.pack(**s.pad())

        self.output_dir = PathEntry(
            s.inner, "output_dir", browse_fn=self._browse_outdir,
            tip="Directory to save output .ply and .obj files")
        self.output_dir.pack(**s.pad())

        self.docker_image = TextEntry(
            s.inner, "Docker Image Name", default="bag-to-mesh",
            tip="Name used when you ran 'docker build -t ...'")
        self.docker_image.pack(**s.pad())

        # Topics
        s2 = Section(parent, "ROS Topics")
        s2.pack(fill="x", padx=8, pady=6)

        self.pc_topic = TextEntry(
            s2.inner, "--pc_topic", default="/points",
            tip="sensor_msgs/PointCloud2 topic name")
        self.pc_topic.pack(**s2.pad())

        self.odom_topic = TextEntry(
            s2.inner, "--odom_topic",
            tip="nav_msgs/Odometry topic -- leave blank to omit")
        self.odom_topic.pack(**s2.pad())

        # Registration
        s3 = Section(parent, "Registration (ICP)")
        s3.pack(fill="x", padx=8, pady=6)

        self.voxel_size = SliderRow(
            s3.inner, "--voxel_size",
            default=0.05, from_=0.001, to=1.0, resolution=0.001,
            tip="Downsampling resolution (m)")
        self.voxel_size.pack(**s3.pad())

        self.icp_dist = SliderRow(
            s3.inner, "--icp_dist_thresh",
            default=0.2, from_=0.01, to=10.0, resolution=0.01,
            tip="Max point correspondence distance (m)")
        self.icp_dist.pack(**s3.pad())

        self.icp_fitness = SliderRow(
            s3.inner, "--icp_fitness_thresh",
            default=0.6, from_=0.0, to=1.0, resolution=0.01,
            tip="Min fraction of points aligned to accept a frame [0-1]")
        self.icp_fitness.pack(**s3.pad())

        self.odom_latency = SliderRow(
            s3.inner, "--odom_max_latency",
            default=0.5, from_=0.0, to=5.0, resolution=0.05,
            tip="Max odom <-> pointcloud timestamp gap (s)")
        self.odom_latency.pack(**s3.pad())

        # Loop Closure
        s4 = Section(parent, "Loop Closure  (disabled by default -- 3-8x slower)")
        s4.pack(fill="x", padx=8, pady=6)

        self.loop_enable = CheckRow(
            s4.inner, "--enable_loop_closure",
            tip="Enable FPFH + RANSAC loop detection")
        self.loop_enable.pack(**s4.pad())
        self.loop_enable.var.trace_add("write",
                                       lambda *_: self._update_command())

        self.loop_radius = SliderRow(
            s4.inner, "--loop_closure_radius",
            default=10.0, from_=1.0, to=50.0, resolution=0.5,
            tip="Pose-space search radius (m)")
        self.loop_radius.pack(**s4.pad())

        self.loop_fitness = SliderRow(
            s4.inner, "--loop_closure_fitness_thresh",
            default=0.3, from_=0.0, to=1.0, resolution=0.01,
            tip="Min ICP fitness to accept a loop constraint [0-1]")
        self.loop_fitness.pack(**s4.pad())

        self.loop_interval = SliderRow(
            s4.inner, "--loop_closure_search_interval",
            default=10, from_=1, to=200, resolution=1, is_int=True,
            tip="Search every N frames")
        self.loop_interval.pack(**s4.pad())

        # Reconstruction
        s5 = Section(parent, "Reconstruction & Post-Processing")
        s5.pack(fill="x", padx=8, pady=6)

        self.poisson_depth = SliderRow(
            s5.inner, "--poisson_depth",
            default=9, from_=6, to=14, resolution=1, is_int=True,
            tip="Octree depth for Poisson meshing (higher = more detail, more RAM)")
        self.poisson_depth.pack(**s5.pad())

        self.density_trim = SliderRow(
            s5.inner, "--density_trim_percentile",
            default=0.05, from_=0.0, to=1.0, resolution=0.01,
            tip="Remove bottom N% of low-density vertices [0-1]")
        self.density_trim.pack(**s5.pad())

        self.level_floor = CheckRow(
            s5.inner, "--level_floor",
            tip="Apply Z-leveling post-processing (flat environments only)")
        self.level_floor.pack(**s5.pad())
        self.level_floor.var.trace_add("write",
                                       lambda *_: self._update_command())

        self.decimate = TextEntry(
            s5.inner, "--decimate_target",
            tip="Ratio <=1 (e.g. 0.25 = keep 25%) or absolute count (e.g. 500000). Blank = skip.")
        self.decimate.pack(**s5.pad())

        # bind all variable traces for live preview
        sliders = (self.voxel_size, self.icp_dist, self.icp_fitness,
                   self.odom_latency, self.loop_radius, self.loop_fitness,
                   self.loop_interval, self.poisson_depth, self.density_trim)
        texts = (self.bag_path, self.output_dir, self.docker_image,
                 self.pc_topic, self.odom_topic, self.decimate)
        for w in sliders:
            w.var.trace_add("write", lambda *_: self._update_command())
        for w in texts:
            w.var.trace_add("write", lambda *_: self._update_command())

    # ── right pane ────────────────────────────────────────────────────────────

    def _build_right(self, parent):
        # Command preview
        cs = Section(parent, "Generated Command")
        cs.pack(fill="x", padx=8, pady=(8, 6))

        cmd_outer = tk.Frame(cs.inner, bg=ENTRY_BG,
                             highlightthickness=1, highlightbackground=BORDER)
        cmd_outer.pack(fill="both")

        # Horizontal scrollbar (packed first so it sticks to bottom)
        xsb = tk.Scrollbar(cmd_outer, orient="horizontal", bg=SURFACE2,
                           troughcolor=BG, activebackground=SURFACE3)
        xsb.pack(side="bottom", fill="x")

        # Vertical scrollbar
        ysb = tk.Scrollbar(cmd_outer, orient="vertical", bg=SURFACE2,
                           troughcolor=BG, activebackground=SURFACE3)
        ysb.pack(side="right", fill="y")

        self.cmd_text = tk.Text(
            cmd_outer, height=9, bg=ENTRY_BG, fg=ACCENT,
            font=FONT_MONO, relief="flat", wrap="none",
            insertbackground=ACCENT, state="disabled", padx=8, pady=6,
            selectbackground=SURFACE3,
            xscrollcommand=xsb.set, yscrollcommand=ysb.set)
        self.cmd_text.pack(side="left", fill="both", expand=True)
        xsb.config(command=self.cmd_text.xview)
        ysb.config(command=self.cmd_text.yview)

        # Log
        ls = Section(parent, "Output Log")
        ls.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        log_frame = tk.Frame(ls.inner, bg=ENTRY_BG,
                             highlightthickness=1, highlightbackground=BORDER)
        log_frame.pack(fill="both", expand=True)
        self.log = scrolledtext.ScrolledText(
            log_frame, bg="#13131f", fg=TEXT,
            font=FONT_MONO, relief="flat", state="disabled",
            wrap="word", padx=8, pady=6, selectbackground=SURFACE3)
        self.log.pack(fill="both", expand=True)

        self.log.tag_config("ok",    foreground=SUCCESS)
        self.log.tag_config("err",   foreground=ERROR)
        self.log.tag_config("warn",  foreground=WARN)
        self.log.tag_config("cmd",   foreground=ACCENT)
        self.log.tag_config("muted", foreground=FAINT)
        self.log.tag_config("info",  foreground=MUTED)

        self._log("Ready. Fill in the parameters and click Run Conversion.", "muted")

    # ── path helpers ──────────────────────────────────────────────────────────

    def _browse_bag(self):
        p = filedialog.askopenfilename(
            title="Select ROS 2 Bag File",
            filetypes=[
                ("ROS 2 bag files", "*.db3 *.bag *.mcap"),
                ("All files", "*.*"),
            ])
        if p:
            self.bag_path.set(p)
            if not self.output_dir.get():
                self.output_dir.set(str(Path(p).parent / "output"))

    def _browse_outdir(self):
        p = filedialog.askdirectory(title="Select Output Directory")
        if p:
            self.output_dir.set(p)

    # ── command builder ───────────────────────────────────────────────────────

    def _build_parts(self):
        """Return (list_of_args, error_string_or_None)."""
        bag = self.bag_path.get()
        out = self.output_dir.get()
        img = self.docker_image.get() or "bag-to-mesh"

        if not bag or not out:
            return None, "bag_path and output_dir are required."

        bag_p = Path(bag)
        out_p = Path(out)

        # Mount bag file's parent dir as /bag, output dir as /output
        parts = [
            "docker", "run", "--rm",
            "-v", f"{bag_p.parent}:/bag",
            "-v", f"{out_p}:/output",
            img,
            f"/bag/{bag_p.name}",
            "/output",
        ]

        # topics
        pc = self.pc_topic.get() or "/points"
        parts += ["--pc_topic", pc]

        odom = self.odom_topic.get()
        if odom:
            parts += ["--odom_topic", odom]

        # registration
        vx = self.voxel_size.get()
        if vx != 0.05:
            parts += ["--voxel_size", str(vx)]

        icp_d = self.icp_dist.get()
        if icp_d != 0.2:
            parts += ["--icp_dist_thresh", str(icp_d)]

        icp_f = self.icp_fitness.get()
        if icp_f != 0.6:
            parts += ["--icp_fitness_thresh", str(icp_f)]

        lat = self.odom_latency.get()
        if lat != 0.5:
            parts += ["--odom_max_latency", str(lat)]

        # loop closure
        if self.loop_enable.get():
            parts += ["--enable_loop_closure"]

            lr = self.loop_radius.get()
            if lr != 10.0:
                parts += ["--loop_closure_radius", str(lr)]

            lf = self.loop_fitness.get()
            if lf != 0.3:
                parts += ["--loop_closure_fitness_thresh", str(lf)]

            li = self.loop_interval.get()
            if li != 10:
                parts += ["--loop_closure_search_interval", str(li)]

        # reconstruction
        pd = self.poisson_depth.get()
        if pd != 9:
            parts += ["--poisson_depth", str(pd)]

        dt = self.density_trim.get()
        if dt != 0.05:
            parts += ["--density_trim_percentile", str(dt)]

        if self.level_floor.get():
            parts += ["--level_floor"]

        dec = self.decimate.get()
        if dec:
            try:
                float(dec)
                parts += ["--decimate_target", dec]
            except ValueError:
                pass

        return parts, None

    @staticmethod
    def _format_cmd(parts):
        """Pretty-print the docker command with backslash line continuations."""
        if not parts:
            return ""
        lines = []
        i = 0
        while i < len(parts):
            tok = parts[i]
            # paired short flags: -v, -e, -p, -u
            if tok in ("-v", "-e", "-p", "-u") and i + 1 < len(parts):
                lines.append(f"  {tok} {shlex.quote(parts[i + 1])}")
                i += 2
            # paired long flags: --something value
            elif (tok.startswith("--")
                  and i + 1 < len(parts)
                  and not parts[i + 1].startswith("-")):
                lines.append(f"  {tok} {shlex.quote(parts[i + 1])}")
                i += 2
            # boolean flag or positional
            else:
                lines.append(tok if i == 0 else f"  {tok}")
                i += 1
        return " \\\n".join(lines)

    def _update_command(self, *_):
        parts, err = self._build_parts()
        text = self._format_cmd(parts) if parts else f"# {err}"
        self.cmd_text.configure(state="normal")
        self.cmd_text.delete("1.0", "end")
        self.cmd_text.insert("end", text)
        self.cmd_text.configure(state="disabled")

    def _copy_cmd(self):
        parts, err = self._build_parts()
        if parts:
            self.clipboard_clear()
            self.clipboard_append(self._format_cmd(parts))
            self._log("Command copied to clipboard.", "muted")
        else:
            messagebox.showwarning("Incomplete", err)

    # ── logging ───────────────────────────────────────────────────────────────

    def _log(self, text, tag=""):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n", tag)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    # ── run ───────────────────────────────────────────────────────────────────

    def _run(self):
        parts, err = self._build_parts()
        if parts is None:
            messagebox.showerror("Missing Input",
                                 "Please specify both bag_path and output_dir.")
            return

        out_p = Path(self.output_dir.get())
        try:
            out_p.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Error",
                                 f"Could not create output directory:\n{e}")
            return

        cmd_str = self._format_cmd(parts)
        self._log(f"\n{'--' * 28}", "muted")
        self._log("$ " + cmd_str, "cmd")
        self._log(f"{'--' * 28}\n", "muted")

        self.run_btn.configure(state="disabled", text="Running...",
                               bg=SURFACE3, fg=MUTED)

        def _worker():
            try:
                proc = subprocess.Popen(
                    parts,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True, bufsize=1)
                for raw_line in proc.stdout:
                    line = raw_line.rstrip()
                    lo = line.lower()
                    if any(k in lo for k in ("error", "traceback", "exception")):
                        tag = "err"
                    elif any(k in lo for k in ("warning", "warn")):
                        tag = "warn"
                    elif any(k in lo for k in ("saved", "done", "complete", "success")):
                        tag = "ok"
                    elif any(k in lo for k in ("registering", "reading",
                                               "optimizing", "merging",
                                               "reconstructing", "extracted")):
                        tag = "info"
                    else:
                        tag = ""
                    self.after(0, self._log, line, tag)
                proc.wait()
                if proc.returncode == 0:
                    self.after(0, self._log,
                               "\nConversion complete!", "ok")
                    self.after(0, self._log,
                               f"   Output: {self.output_dir.get()}", "ok")
                else:
                    self.after(0, self._log,
                               f"\nProcess exited with code {proc.returncode}", "err")
            except FileNotFoundError:
                self.after(0, self._log,
                           "docker not found -- is Docker installed and on PATH?", "err")
            except Exception as exc:
                self.after(0, self._log, f"Unexpected error: {exc}", "err")
            finally:
                self.after(0, lambda: self.run_btn.configure(
                    state="normal", text="Run Conversion",
                    bg=ACCENT, fg="#1e1e2e"))

        threading.Thread(target=_worker, daemon=True).start()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()
