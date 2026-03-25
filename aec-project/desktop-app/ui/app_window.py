"""
Desktop App UI — tkinter

Simple but functional interface:
    - Connection status (connected / waiting)
    - AEC ON/OFF toggle button (green/red)
    - Live ERLE meter (dB) — updates every 2 seconds
    - Packet stats (sent / received / lost)
    - Peer IP:Port display
"""

import tkinter as tk
from tkinter import ttk
import threading
import time
import queue
from typing import Callable


class AppWindow:
    """
    Main application window.

    Communicates with the backend via:
        - on_aec_toggle(enabled: bool) callback
        - metrics_queue: Queue where backend pushes periodic metric dicts
    """

    def __init__(
        self,
        peer_addr:      str,
        local_port:     int,
        on_aec_toggle:  Callable[[bool], None],
        metrics_queue:  queue.Queue,
    ) -> None:
        self._peer_addr     = peer_addr
        self._local_port    = local_port
        self._on_aec_toggle = on_aec_toggle
        self._metrics_queue = metrics_queue
        self._aec_enabled   = False

        self._root = tk.Tk()
        self._root.title("AEC Voice Call")
        self._root.geometry("420x320")
        self._root.resizable(False, False)
        self._root.configure(bg="#1e1e2e")

        self._build_ui()
        self._start_metrics_poll()

    # ------------------------------------------------------------------ #
    # UI Construction
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        root = self._root
        PAD = {"padx": 12, "pady": 6}
        BG  = "#1e1e2e"
        FG  = "#cdd6f4"
        ACCENT = "#89b4fa"

        # Title
        tk.Label(root, text="AEC Voice Call", font=("Helvetica", 16, "bold"),
                 bg=BG, fg=ACCENT).pack(pady=(16, 4))

        # Connection info
        info_frame = tk.Frame(root, bg=BG)
        info_frame.pack(fill="x", **PAD) # type: ignore
        tk.Label(info_frame, text=f"Local port: {self._local_port}",
                 font=("Courier", 10), bg=BG, fg=FG).pack(anchor="w")
        tk.Label(info_frame, text=f"Peer: {self._peer_addr}",
                 font=("Courier", 10), bg=BG, fg=FG).pack(anchor="w")

        # Status indicator
        self._status_var = tk.StringVar(value="● Connecting...")
        tk.Label(root, textvariable=self._status_var,
                 font=("Helvetica", 11), bg=BG, fg="#f38ba8").pack(pady=4)

        # Separator
        ttk.Separator(root, orient="horizontal").pack(fill="x", padx=12, pady=8)

        # AEC Toggle Button
        self._aec_btn = tk.Button(
            root,
            text="AEC: OFF",
            font=("Helvetica", 14, "bold"),
            width=16,
            bg="#313244",
            fg="#f38ba8",
            activebackground="#45475a",
            relief="flat",
            bd=0,
            cursor="hand2",
            command=self._toggle_aec,
        )
        self._aec_btn.pack(pady=8)

        # ERLE meter
        metrics_frame = tk.Frame(root, bg=BG)
        metrics_frame.pack(fill="x", **PAD) # type: ignore

        tk.Label(metrics_frame, text="ERLE:", font=("Courier", 11, "bold"),
                 bg=BG, fg=FG, width=14, anchor="w").grid(row=0, column=0)
        self._erle_var = tk.StringVar(value="—")
        tk.Label(metrics_frame, textvariable=self._erle_var,
                 font=("Courier", 11), bg=BG, fg="#a6e3a1", width=12).grid(row=0, column=1)

        tk.Label(metrics_frame, text="Filter norm:",
                 font=("Courier", 11, "bold"), bg=BG, fg=FG, width=14, anchor="w").grid(row=1, column=0)
        self._norm_var = tk.StringVar(value="—")
        tk.Label(metrics_frame, textvariable=self._norm_var,
                 font=("Courier", 11), bg=BG, fg=FG, width=12).grid(row=1, column=1)

        tk.Label(metrics_frame, text="Delay est.:",
                 font=("Courier", 11, "bold"), bg=BG, fg=FG, width=14, anchor="w").grid(row=2, column=0)
        self._delay_var = tk.StringVar(value="—")
        tk.Label(metrics_frame, textvariable=self._delay_var,
                 font=("Courier", 11), bg=BG, fg=FG, width=12).grid(row=2, column=1)

        tk.Label(metrics_frame, text="Double-talk:",
                 font=("Courier", 11, "bold"), bg=BG, fg=FG, width=14, anchor="w").grid(row=3, column=0)
        self._dt_var = tk.StringVar(value="—")
        tk.Label(metrics_frame, textvariable=self._dt_var,
                 font=("Courier", 11), bg=BG, fg=FG, width=12).grid(row=3, column=1)

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #

    def _toggle_aec(self) -> None:
        self._aec_enabled = not self._aec_enabled
        self._on_aec_toggle(self._aec_enabled)

        if self._aec_enabled:
            self._aec_btn.config(text="AEC: ON", bg="#1e6641", fg="#a6e3a1")
            self._erle_var.set("adapting...")
        else:
            self._aec_btn.config(text="AEC: OFF", bg="#313244", fg="#f38ba8")
            self._erle_var.set("—")
            self._norm_var.set("—")
            self._delay_var.set("—")
            self._dt_var.set("—")

    def set_connected(self, connected: bool) -> None:
        """Called by main thread when first packet arrives."""
        def _update():
            if connected:
                self._status_var.set("● Connected")
                # Can't directly access _status_label fg, use workaround
                for widget in self._root.winfo_children():
                    if isinstance(widget, tk.Label) and "Connecting" in (widget.cget("text") or ""):
                        widget.config(fg="#a6e3a1")
        self._root.after(0, _update)

    # ------------------------------------------------------------------ #
    # Metrics polling
    # ------------------------------------------------------------------ #

    def _start_metrics_poll(self) -> None:
        """Poll metrics_queue periodically and update UI labels."""
        self._poll_metrics()

    def _poll_metrics(self) -> None:
        try:
            while True:
                metrics = self._metrics_queue.get_nowait()
                self._update_metrics(metrics)
        except queue.Empty:
            pass
        # Re-schedule
        self._root.after(500, self._poll_metrics)

    def _update_metrics(self, metrics: dict) -> None:
        erle = metrics.get("erle_db", None)
        if erle is not None:
            color = "#a6e3a1" if erle >= 15 else "#f9e2af" if erle >= 8 else "#f38ba8"
            self._erle_var.set(f"{erle:.1f} dB")
        norm = metrics.get("filter_norm")
        if norm is not None:
            self._norm_var.set(f"{norm:.3f}")
        delay = metrics.get("delay_ms")
        if delay is not None:
            self._delay_var.set(f"{delay:.1f} ms")
        dt = metrics.get("double_talk_ratio")
        if dt is not None:
            self._dt_var.set(f"{dt*100:.1f}%")

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        """Block until window is closed."""
        self._root.mainloop()