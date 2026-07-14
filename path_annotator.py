import tkinter as tk
from tkinter import messagebox
import json
from enum import Enum, auto
from typing import List, Tuple, Dict, Any

# ==========================================
# Functional Core (Pure Functions)
# ==========================================
# These functions handle data transformation without side effects.

def to_cartesian(vertices: List[Tuple[float, float]], canvas_height: float) -> List[Tuple[float, float]]:
    """
    Transforms Tkinter canvas coordinates (top-left origin, Y-down) 
    to standard Cartesian coordinates (bottom-left origin, Y-up).
    """
    return [(x, canvas_height - y) for x, y in vertices]


def calculate_signed_area(vertices: List[Tuple[float, float]]) -> float:
    """
    Calculates the signed area of a 2D polygon using the Shoelace formula.
    
    Design Intent:
    Because the coordinates are transformed to standard Cartesian space (Y-up) 
    before this function is called, a positive signed area natively indicates 
    counter-clockwise (CCW) winding.
    """
    area = 0.0
    n = len(vertices)
    if n < 3:
        return 0.0

    for i in range(n):
        j = (i + 1) % n
        x1, y1 = vertices[i]
        x2, y2 = vertices[j]
        area += (x1 * y2 - x2 * y1)
        
    return area / 2.0


def ensure_ccw(vertices: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Validates the winding order of a polygon and reverses it if it is clockwise.
    Returns a new list of vertices to ensure immutability of the input data.
    """
    if len(vertices) < 3:
        return list(vertices)
        
    area = calculate_signed_area(vertices)
    # If area is negative, winding is Clockwise. Return reversed list for CCW.
    if area < 0:
        return list(reversed(vertices))
        
    return list(vertices)


def generate_export_payload(
    boundary: List[Tuple[float, float]], 
    obstacles: List[List[Tuple[float, float]]], 
    path: List[Tuple[float, float]],
    canvas_height: float
) -> Dict[str, Any]:
    """
    Transforms coordinates to bottom-left origin and assembles the final 
    data structure, applying required formatting constraints:
    - Origin: Bottom-Left (Y-up)
    - Boundary: Counter-Clockwise (CCW)
    - Obstacles: Counter-Clockwise (CCW)
    - Path: Exact user input order
    """
    # 1. Transform all points to Cartesian coordinates (bottom-left origin)
    cartesian_boundary = to_cartesian(boundary, canvas_height)
    cartesian_obstacles = [to_cartesian(obs, canvas_height) for obs in obstacles]
    cartesian_path = to_cartesian(path, canvas_height)

    # 2. Enforce winding constraints on polygons
    return {
        "boundary": ensure_ccw(cartesian_boundary),
        "obstacles": [ensure_ccw(obs) for obs in cartesian_obstacles],
        "path": cartesian_path  # Preserves user input order
    }


# ==========================================
# State and GUI Management
# ==========================================

class DrawingMode(Enum):
    BOUNDARY = auto()
    OBSTACLES = auto()
    PATH = auto()


class PathSmoothingGUI(tk.Tk):
    """
    Main GUI application class. Manages the event loop, state transitions,
    and rendering logic. Exclusively handles side-effects.
    """
    def __init__(self):
        super().__init__()
        self.title("Path Smoothing Environment Annotator")
        self.geometry("900x700")
        
        # --- Application State ---
        self.mode = DrawingMode.BOUNDARY
        
        self.boundary: List[Tuple[float, float]] = []
        self.obstacles: List[List[Tuple[float, float]]] = []
        self.noisy_path: List[Tuple[float, float]] = []
        
        self.current_vertices: List[Tuple[float, float]] = []
        
        self._setup_ui()
        self._update_status()

    def _setup_ui(self) -> None:
        """Initializes the Tkinter widgets and layout."""
        # Main layout containers
        self.canvas = tk.Canvas(self, bg="white", cursor="cross")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        self.panel = tk.Frame(self, width=250)
        self.panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        # UI Controls
        self.status_label = tk.Label(self.panel, text="", font=("Arial", 12, "bold"), fg="blue", wraplength=200)
        self.status_label.pack(pady=20)

        self.btn_finish_shape = tk.Button(self.panel, text="Close & Finish Shape", command=self.finish_current_shape)
        self.btn_finish_shape.pack(fill=tk.X, pady=5)

        self.btn_next_mode = tk.Button(self.panel, text="Next Mode (Obstacles)", command=self.advance_mode)
        self.btn_next_mode.pack(fill=tk.X, pady=5)

        self.btn_export = tk.Button(self.panel, text="Confirm & Export JSON", command=self.export_data, bg="lightgreen")
        self.btn_export.pack(fill=tk.X, pady=30)
        
        self.text_output = tk.Text(self.panel, height=15, width=30, state=tk.DISABLED)
        self.text_output.pack(fill=tk.BOTH, expand=True)

    def _update_status(self) -> None:
        """Updates the instructional text based on the current state machine mode."""
        if self.mode == DrawingMode.BOUNDARY:
            self.status_label.config(
                text="Mode: BOUNDARY\nClick to draw the environment boundary. Click 'Close & Finish' when done."
            )
            self.btn_next_mode.config(state=tk.DISABLED)
        elif self.mode == DrawingMode.OBSTACLES:
            self.status_label.config(
                text="Mode: OBSTACLES\nDraw an obstacle. Click 'Close & Finish' per obstacle."
            )
            self.btn_next_mode.config(state=tk.NORMAL, text="Next Mode (Path)")
        elif self.mode == DrawingMode.PATH:
            self.status_label.config(
                text="Mode: PATH\nDraw the noisy path. Click 'Confirm & Export' when done."
            )
            self.btn_finish_shape.config(state=tk.DISABLED)
            self.btn_next_mode.config(state=tk.DISABLED)

    # --- Event Handlers & State Transitions ---

    def on_canvas_click(self, event: tk.Event) -> None:
        """Handles mouse clicks, appending points and rendering immediate visual feedback."""
        pt = (float(event.x), float(event.y))
        
        # Functional style update: creating a new list rather than mutating in place
        self.current_vertices = self.current_vertices + [pt]
        
        # Render visual point
        r = 3
        color = {
            DrawingMode.BOUNDARY: "black",
            DrawingMode.OBSTACLES: "red",
            DrawingMode.PATH: "blue"
        }[self.mode]
        
        self.canvas.create_oval(pt[0]-r, pt[1]-r, pt[0]+r, pt[1]+r, fill=color, outline=color)
        
        # Render line to previous point
        if len(self.current_vertices) > 1:
            prev_pt = self.current_vertices[-2]
            self.canvas.create_line(prev_pt[0], prev_pt[1], pt[0], pt[1], fill=color, width=2)

    def finish_current_shape(self) -> None:
        """Closes the current polygon, stores it in application state, and resets the buffer."""
        if len(self.current_vertices) < 3:
            messagebox.showwarning("Warning", "A polygon requires at least 3 points.")
            return

        # Visually close the polygon
        first_pt = self.current_vertices[0]
        last_pt = self.current_vertices[-1]
        color = "black" if self.mode == DrawingMode.BOUNDARY else "red"
        self.canvas.create_line(last_pt[0], last_pt[1], first_pt[0], first_pt[1], fill=color, width=2)

        # State dispatch based on current mode
        if self.mode == DrawingMode.BOUNDARY:
            self.boundary = list(self.current_vertices)
            self.advance_mode()
        elif self.mode == DrawingMode.OBSTACLES:
            # Append new immutable copy to obstacles
            self.obstacles = self.obstacles + [list(self.current_vertices)]
        
        # Reset current buffer
        self.current_vertices = []

    def advance_mode(self) -> None:
        """Transitions the application state machine to the next drawing phase."""
        if self.mode == DrawingMode.BOUNDARY:
            self.mode = DrawingMode.OBSTACLES
        elif self.mode == DrawingMode.OBSTACLES:
            self.mode = DrawingMode.PATH
            
        self.current_vertices = []
        self._update_status()

    def export_data(self) -> None:
        """Finalizes input, delegates to pure functions for formatting, and outputs JSON."""
        # If in PATH mode, the current vertices represent the path.
        if self.mode == DrawingMode.PATH:
            if len(self.current_vertices) < 2:
                messagebox.showerror("Error", "Path must have at least 2 points.")
                return
            self.noisy_path = list(self.current_vertices)

        if not self.boundary:
            messagebox.showerror("Error", "Boundary is missing.")
            return

        # Retrieve dynamic canvas height for origin transformation
        canvas_height = float(self.canvas.winfo_height())

        # Execute pure functions to transform coordinates and guarantee winding constraints
        payload = generate_export_payload(
            self.boundary, 
            self.obstacles, 
            self.noisy_path, 
            canvas_height
        )
        json_str = json.dumps(payload, indent=2)

        # Output side-effects
        self.text_output.config(state=tk.NORMAL)
        self.text_output.delete("1.0", tk.END)
        self.text_output.insert(tk.END, json_str)
        self.text_output.config(state=tk.DISABLED)
        
        print("--- Exported Environment JSON ---")
        print(json_str)
        print("---------------------------------")


if __name__ == "__main__":
    app = PathSmoothingGUI()
    app.mainloop()
