import sys
import numpy as np
from collections import deque
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QPushButton, QSlider, QLabel, QComboBox, 
    QLineEdit, QFormLayout, QGroupBox, QCheckBox, QFileDialog,
    QColorDialog
)
from PySide6.QtCore import Qt, QTimer, QPoint
from PySide6.QtGui import QPainter, QColor, QImage, QPen, QPolygon, QBrush

class CAWidget(QWidget):
    def __init__(self, width=300, height=300, cell_size=2):
        super().__init__()
        self.grid_w = width
        self.grid_h = height
        self.cell_size = cell_size
        
        self.age_grid = np.zeros((height, width), dtype=np.uint8)
        self.next_age_grid = np.zeros((height, width), dtype=np.uint8)
        self.color_grid = np.zeros((height, width, 3), dtype=np.uint8)
        self.next_color_grid = np.zeros((height, width, 3), dtype=np.uint8)
        
        self.max_age = 60
        self.mutation_rate = 0.01
        self.wrap_edges = True
        self.use_von_neumann = False
        self.show_grid_lines = False
        self.view_mode = "GENETIC"  # "GENETIC" or "HEATMAP"
        
        self.capacity = 1.0  # 1.0 = 100% of grid
        
        self.brush_size = 1
        self.draw_color = np.array([0, 255, 200], dtype=np.uint8) # Cyan
        
        self.setFixedSize(width * cell_size, height * cell_size)
        self.setMouseTracking(True)
        
        self.birth_rules = np.array([3, 6], dtype=np.uint8)
        self.survive_rules = np.array([2, 3], dtype=np.uint8)
        
        self.drawing = False
        self.erase_mode = False
        self.stamp_mode = False
        
        self.patterns = {
            "HighLife Replicator": ["...OO...", "..O..O..", ".O....O.", "OO.OO.OO"],
            "Fredkin Cross": ["..O..", "..O..", "OOOOO", "..O..", "..O.."],
            "Glider": [".O.", "..O", "OOO"],
            "Acorn": [".O.....", "...O...", "OO..OOO"],
            "R-Pentomino": [".OO", "OO.", ".O."]
        }
        self.current_pattern_name = "HighLife Replicator"
        self.load_pattern("HighLife Replicator")

    def update_cell_size(self, size):
        self.cell_size = size
        self.setFixedSize(self.grid_w * size, self.grid_h * size)
        self.update()

    def clear_grid(self):
        self.age_grid.fill(0)
        self.color_grid.fill(0)
        self.update()

    def load_pattern(self, pattern_name, center_x=None, center_y=None):
        if center_x is None: self.clear_grid()
        if pattern_name == "Random Soup":
            mask = np.random.rand(self.grid_h, self.grid_w) < 0.3
            self.age_grid[mask] = 1
            self.color_grid[mask] = np.random.randint(0, 255, (np.sum(mask), 3), dtype=np.uint8)
            self.update()
            return
            
        if pattern_name in self.patterns:
            pattern = self.patterns[pattern_name]
            h = len(pattern)
            w = len(pattern[0])
            
            if center_x is None: center_x = self.grid_w // 2
            if center_y is None: center_y = self.grid_h // 2
            
            start_x = center_x - w // 2
            start_y = center_y - h // 2
            
            for y, row in enumerate(pattern):
                for x, char in enumerate(row):
                    if char == 'O':
                        px, py = start_x + x, start_y + y
                        if 0 <= px < self.grid_w and 0 <= py < self.grid_h:
                            self.age_grid[py, px] = 1
                            self.color_grid[py, px] = self.draw_color
            self.update()

    def set_rules(self, birth_str, survive_str):
        try:
            self.birth_rules = np.array([int(c) for c in birth_str if c.isdigit()], dtype=np.uint8)
            self.survive_rules = np.array([int(c) for c in survive_str if c.isdigit()], dtype=np.uint8)
            return True
        except Exception:
            return False

    def _get_neighbor_data(self, bool_grid, color_grid):
        # Select topology
        if self.use_von_neumann:
            # 4 Neighbors (Up, Down, Left, Right)
            shifts = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        else:
            # 8 Neighbors (Moore)
            shifts = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
            
        n_bools = []
        n_colors = []
        
        for dy, dx in shifts:
            if self.wrap_edges:
                n_bools.append(np.roll(bool_grid, dy, axis=0))
                n_bools.append(np.roll(bool_grid, dx, axis=1)) # Wait, np.roll takes (shift, axis)
                # Correction for combined shifts:
                # Actually, it's easier to do:
                pass
            else:
                # Bounded
                pass

        # Let's rewrite the roll/pad logic clearly
        n_bools = []
        n_colors = []
        
        # Mask colors so dead neighbors don't contribute
        masked_colors = (color_grid * bool_grid[..., None]).astype(np.int16)
        
        for dy, dx in shifts:
            if self.wrap_edges:
                rb = np.roll(bool_grid, dy, axis=0)
                rb = np.roll(rb, dx, axis=1)
                rc = np.roll(masked_colors, dy, axis=0)
                rc = np.roll(rc, dx, axis=1)
            else:
                rb = np.pad(bool_grid, 1, mode='constant', constant_values=0)
                rb = rb[1+dy:1+dy+self.grid_h, 1+dx:1+dx+self.grid_w]
                
                rc = np.pad(masked_colors, 1, mode='constant', constant_values=0)
                rc = rc[1+dy:1+dy+self.grid_h, 1+dx:1+dx+self.grid_w]
                
            n_bools.append(rb)
            n_colors.append(rc)
            
        return sum(n_bools), sum(n_colors)

    def step(self):
        bool_grid = (self.age_grid > 0).astype(np.uint8)
        neighbors, color_sums = self._get_neighbor_data(bool_grid, self.color_grid)
        
        birth_mask = np.isin(neighbors, self.birth_rules) & (self.age_grid == 0)
        survive_mask = np.isin(neighbors, self.survive_rules) & (self.age_grid > 0)
        
        self.next_age_grid.fill(0)
        self.next_age_grid[birth_mask] = 1
        self.next_age_grid[survive_mask] = np.minimum(self.age_grid[survive_mask] + 1, self.max_age)
        
        # Ecological Carrying Capacity (Resource Starvation)
        total_alive = np.sum(self.next_age_grid > 0)
        max_capacity = int(self.capacity * self.grid_w * self.grid_h)
        
        if total_alive > max_capacity:
            # Randomly kill off the excess population
            overpop = total_alive - max_capacity
            alive_indices = np.argwhere(self.next_age_grid > 0)
            kill_indices = np.random.choice(len(alive_indices), size=overpop, replace=False)
            for idx in kill_indices:
                y, x = alive_indices[idx]
                self.next_age_grid[y, x] = 0
        
        # Blended Genetic Inheritance
        self.next_color_grid.fill(0)
        self.next_color_grid[survive_mask] = self.color_grid[survive_mask]
        
        safe_neighbors = np.where(neighbors == 0, 1, neighbors)
        avg_parents = (color_sums // safe_neighbors[..., None]).astype(np.uint8)
        
        mut_mask = (np.random.rand(self.grid_h, self.grid_w) < self.mutation_rate) & birth_mask
        if np.any(mut_mask):
            random_colors = np.random.randint(50, 255, (np.sum(mut_mask), 3), dtype=np.uint8)
            avg_parents[mut_mask] = random_colors
            
        self.next_color_grid[birth_mask] = avg_parents[birth_mask]
        
        self.age_grid, self.next_age_grid = self.next_age_grid, self.age_grid
        self.color_grid, self.next_color_grid = self.next_color_grid, self.color_grid
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(10, 10, 15))
        
        render_colors = np.zeros((self.grid_h, self.grid_w, 3), dtype=np.uint8)
        
        if self.view_mode == "GENETIC":
            age_factor = np.ones_like(self.age_grid, dtype=np.float32)
            age_factor[self.age_grid >= 2] = 1.0 - ((self.age_grid[self.age_grid >= 2] - 2) / self.max_age) * 0.8
            render_colors = (self.color_grid * age_factor[..., None]).astype(np.uint8)
            render_colors[self.age_grid == 1] = [255, 255, 255] # Birth flash
        elif self.view_mode == "HEATMAP":
            # Map Age to a Heatmap LUT
            # 1 = White/Yellow, 30 = Red, 60 = Dark Purple
            for age in range(1, self.max_age + 1):
                t = age / self.max_age
                if t < 0.3:
                    # White to Yellow to Red
                    r = 255
                    g = int(255 - (255 * (t / 0.3)))
                    b = int(255 - (255 * (t / 0.3)))
                else:
                    # Red to Dark Purple
                    r = int(255 - (155 * ((t - 0.3) / 0.7)))
                    g = 0
                    b = int(0 + (100 * ((t - 0.3) / 0.7)))
                render_colors[self.age_grid == age] = [r, g, b]
                
        render_colors[self.age_grid == 0] = [10, 10, 15]
        
        self._render_buffer = render_colors 
        image = QImage(self._render_buffer.data, self.grid_w, self.grid_h, self.grid_w * 3, QImage.Format.Format_RGB888)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        painter.drawImage(self.rect(), image)
        
        if self.show_grid_lines and self.cell_size >= 4:
            painter.setPen(QPen(QColor(255, 255, 255, 30), 1))
            for x in range(0, self.width(), self.cell_size):
                painter.drawLine(x, 0, x, self.height())
            for y in range(0, self.height(), self.cell_size):
                painter.drawLine(0, y, self.width(), y)

    def mousePressEvent(self, event):
        self.drawing = True
        self.erase_mode = (event.button() == Qt.MouseButton.RightButton)
        self._handle_mouse(event.position().toPoint(), is_press=True)

    def mouseMoveEvent(self, event):
        if self.drawing:
            self._handle_mouse(event.position().toPoint(), is_press=False)

    def mouseReleaseEvent(self, event):
        self.drawing = False

    def _handle_mouse(self, pos, is_press):
        x = int(pos.x() / self.cell_size)
        y = int(pos.y() / self.cell_size)
        
        if self.stamp_mode and is_press and not self.erase_mode:
            self.load_pattern(self.current_pattern_name, center_x=x, center_y=y)
        else:
            r = self.brush_size // 2
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    px, py = x + dx, y + dy
                    if 0 <= px < self.grid_w and 0 <= py < self.grid_h:
                        self.age_grid[py, px] = 0 if self.erase_mode else 1
                        self.color_grid[py, px] = self.draw_color
            self.update()

    def count_alive(self):
        return int(np.count_nonzero(self.age_grid > 0))
        
    def get_average_color(self):
        alive_mask = self.age_grid > 0
        if np.any(alive_mask):
            avg = np.mean(self.color_grid[alive_mask], axis=0)
            return QColor(int(avg[0]), int(avg[1]), int(avg[2]))
        return QColor(80, 80, 80)

    def get_genetic_diversity(self):
        alive_mask = self.age_grid > 0
        if np.sum(alive_mask) < 10: return 0.0
        variance = np.var(self.color_grid[alive_mask], axis=0)
        return min(100.0, np.mean(variance) / 160.0)

    def save_image(self, filepath):
        image = QImage(self.color_grid.data, self.grid_w, self.grid_h, self.grid_w * 3, QImage.Format.Format_RGB888)
        image.save(filepath)


class GraphWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(120)
        self.history = deque(maxlen=200)
        self.max_pop = 1

    def add_data(self, pop, diversity, color: QColor):
        self.history.append((pop, diversity, color))
        if pop > self.max_pop: self.max_pop = pop
        self.update()

    def clear(self):
        self.history.clear()
        self.max_pop = 1
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(20, 20, 30))
        if len(self.history) < 2: return
            
        w, h = self.width(), self.height()
        padding = 10
        
        painter.setPen(QColor(80, 80, 80))
        painter.drawLine(padding, h - padding, w - padding, h - padding)
        painter.drawLine(padding, padding, padding, h - padding)
        
        pop_points = []
        div_points = []
        for i, (pop, div, color) in enumerate(self.history):
            x = padding + (i / (self.history.maxlen - 1)) * (w - 2 * padding)
            y_pop = (h - padding) - (pop / self.max_pop) * (h - 2 * padding)
            y_div = (h - padding) - (div / 100.0) * (h - 2 * padding)
            pop_points.append((x, y_pop, color))
            div_points.append((x, y_div))
            
        poly = QPolygon([QPoint(int(p[0]), int(p[1])) for p in pop_points])
        poly.append(QPoint(int(pop_points[-1][0]), h - padding))
        poly.append(QPoint(int(pop_points[0][0]), h - padding))
        fill_color = QColor(self.history[-1][2])
        fill_color.setAlpha(40)
        painter.setBrush(QBrush(fill_color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPolygon(poly)
        
        for i in range(len(pop_points) - 1):
            painter.setPen(QPen(pop_points[i][2], 2))
            painter.drawLine(int(pop_points[i][0]), int(pop_points[i][1]), 
                             int(pop_points[i+1][0]), int(pop_points[i+1][1]))
                             
        painter.setPen(QPen(QColor(255, 0, 255, 200), 2))
        for i in range(len(div_points) - 1):
            painter.drawLine(int(div_points[i][0]), int(div_points[i][1]), 
                             int(div_points[i+1][0]), int(div_points[i+1][1]))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quantum Evolution Lab: Ecological & Fractal Dynamics")
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        top_layout = QHBoxLayout()
        
        # --- LEFT PANEL (Grouped Controls) ---
        left_scroll_layout = QVBoxLayout()
        
        # Group 1: Simulation Control
        sim_group = QGroupBox("Simulation Control")
        sim_layout = QVBoxLayout(sim_group)
        btn_row1 = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_step = QPushButton("Step")
        self.btn_rewind = QPushButton("Rewind")
        btn_row1.addWidget(self.btn_start); btn_row1.addWidget(self.btn_step); btn_row1.addWidget(self.btn_rewind)
        
        btn_row2 = QHBoxLayout()
        self.btn_clear = QPushButton("Clear")
        self.btn_save = QPushButton("Save PNG")
        self.btn_csv = QPushButton("Export CSV")
        btn_row2.addWidget(self.btn_clear); btn_row2.addWidget(self.btn_save); btn_row2.addWidget(self.btn_csv)
        
        self.turbo_check = QCheckBox("Turbo Mode (5x Speed)")
        
        sim_layout.addLayout(btn_row1)
        sim_layout.addLayout(btn_row2)
        sim_layout.addWidget(self.turbo_check)
        left_scroll_layout.addWidget(sim_group)
        
        # Group 2: Environment & View
        env_group = QGroupBox("Environment & View")
        env_layout = QVBoxLayout(env_group)
        
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(["HighLife Replicator", "Fredkin Cross", "Glider", "Acorn", "R-Pentomino", "Random Soup"])
        self.pattern_combo.currentTextChanged.connect(self.change_pattern)
        
        self.stamp_check = QCheckBox("Stamp Mode (Left-click to paste)")
        self.wrap_check = QCheckBox("Wrap Edges (Torus Topology)")
        self.wrap_check.setChecked(True)
        self.vn_check = QCheckBox("Von Neumann Neighborhood (4-Neighbor)")
        self.grid_check = QCheckBox("Show Grid Lines")
        self.heat_check = QCheckBox("Thermal Heatmap View (Age)")
        
        env_layout.addWidget(QLabel("Load Pattern:"))
        env_layout.addWidget(self.pattern_combo)
        env_layout.addWidget(self.stamp_check)
        env_layout.addWidget(self.wrap_check)
        env_layout.addWidget(self.vn_check)
        env_layout.addWidget(self.grid_check)
        env_layout.addWidget(self.heat_check)
        left_scroll_layout.addWidget(env_group)
        
        # Group 3: Evolution & DNA
        evo_group = QGroupBox("Evolution & DNA")
        evo_layout = QVBoxLayout(evo_group)
        
        self.btn_color = QPushButton("Choose Draw Color (DNA)")
        self.btn_color.setStyleSheet("background-color: #00ffcc; color: black;")
        
        evo_layout.addWidget(self.btn_color)
        evo_layout.addWidget(QLabel("Mutation Rate (%):"))
        self.mut_slider = QSlider(Qt.Orientation.Horizontal)
        self.mut_slider.setRange(0, 100); self.mut_slider.setValue(1)
        
        evo_layout.addWidget(QLabel("Carrying Capacity (Max Pop %):"))
        self.cap_slider = QSlider(Qt.Orientation.Horizontal)
        self.cap_slider.setRange(10, 100); self.cap_slider.setValue(100)
        
        evo_layout.addWidget(self.mut_slider)
        evo_layout.addWidget(self.cap_slider)
        left_scroll_layout.addWidget(evo_group)
        
        # Group 4: Rules & Tools
        rules_group = QGroupBox("Rules & Tools")
        rules_layout = QVBoxLayout(rules_group)
        
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["HighLife (B36/S23)", "Conway Life (B3/S23)", "Fredkin (B1357/S02468)", "Seeds (B2/S)"])
        self.preset_combo.currentTextChanged.connect(self.change_preset)
        
        rules_form = QFormLayout()
        self.birth_input = QLineEdit("36")
        self.survive_input = QLineEdit("23")
        rules_form.addRow("Birth (B):", self.birth_input)
        rules_form.addRow("Survive (S):", self.survive_input)
        
        self.btn_apply_rules = QPushButton("Apply Custom Rules")
        
        rules_layout.addWidget(QLabel("Ruleset Preset:"))
        rules_layout.addWidget(self.preset_combo)
        rules_layout.addLayout(rules_form)
        rules_layout.addWidget(self.btn_apply_rules)
        left_scroll_layout.addWidget(rules_group)
        
        # Group 5: Render Settings
        render_group = QGroupBox("Render Settings")
        render_layout = QVBoxLayout(render_group)
        render_layout.addWidget(QLabel("Simulation Speed:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(10, 200); self.speed_slider.setValue(40)
        
        render_layout.addWidget(QLabel("Zoom (Cell Size):"))
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(1, 8); self.zoom_slider.setValue(2)
        
        render_layout.addWidget(QLabel("Brush Size:"))
        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setRange(1, 7); self.brush_slider.setValue(1)
        
        render_layout.addWidget(self.speed_slider)
        render_layout.addWidget(self.zoom_slider)
        render_layout.addWidget(self.brush_slider)
        left_scroll_layout.addWidget(render_group)
        
        # Stats Labels
        self.lbl_generation = QLabel("Generation: 0")
        self.lbl_population = QLabel("Population: 0")
        self.lbl_diversity = QLabel("Diversity: 0.0%")
        self.lbl_generation.setStyleSheet("font-size: 14px; font-weight: bold; color: #00ffcc;")
        self.lbl_population.setStyleSheet("font-size: 14px; font-weight: bold; color: #00ffcc;")
        self.lbl_diversity.setStyleSheet("font-size: 14px; font-weight: bold; color: #ff00ff;")
        left_scroll_layout.addWidget(self.lbl_generation)
        left_scroll_layout.addWidget(self.lbl_population)
        left_scroll_layout.addWidget(self.lbl_diversity)
        
        # Add stretch to push everything up
        left_scroll_layout.addStretch(1)
        
        # Connect Signals
        self.btn_start.clicked.connect(self.toggle_simulation)
        self.btn_step.clicked.connect(self.step_once)
        self.btn_rewind.clicked.connect(self.rewind)
        self.btn_clear.clicked.connect(self.ca_clear)
        self.btn_save.clicked.connect(self.save_image)
        self.btn_csv.clicked.connect(self.export_csv)
        self.btn_color.clicked.connect(self.choose_color)
        self.btn_apply_rules.clicked.connect(self.apply_rules)
        
        self.stamp_check.stateChanged.connect(self.toggle_stamp)
        self.wrap_check.stateChanged.connect(self.toggle_wrap)
        self.vn_check.stateChanged.connect(self.toggle_vn)
        self.turbo_check.stateChanged.connect(self.toggle_turbo)
        self.grid_check.stateChanged.connect(self.toggle_grid)
        self.heat_check.stateChanged.connect(self.toggle_heatmap)
        
        self.speed_slider.valueChanged.connect(self.update_speed)
        self.zoom_slider.valueChanged.connect(self.update_zoom)
        self.brush_slider.valueChanged.connect(self.update_brush)
        self.mut_slider.valueChanged.connect(self.update_mutation)
        self.cap_slider.valueChanged.connect(self.update_capacity)
        
        top_layout.addLayout(left_scroll_layout, 1)
        
        # --- RIGHT PANEL (Grid) ---
        self.ca_widget = CAWidget(width=300, height=300, cell_size=2)
        top_layout.addWidget(self.ca_widget, 2)
        main_layout.addLayout(top_layout, 3)
        
        # --- BOTTOM PANEL (Graph) ---
        graph_group = QGroupBox("Ecosystem Dynamics (Cyan=Population, Magenta=Genetic Diversity)")
        graph_layout = QVBoxLayout(graph_group)
        self.graph_widget = GraphWidget()
        graph_layout.addWidget(self.graph_widget)
        main_layout.addWidget(graph_group, 1)
        
        # Timer & State
        self.timer = QTimer()
        self.timer.timeout.connect(self.step_and_update)
        self.is_running = False
        self.generation = 0
        self.history = deque(maxlen=100)
        self.turbo_mode = False
        self.csv_history = [] # For CSV export
        self.update_speed()
        self.update_mutation(1)
        
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; }
            QGroupBox { color: #aaaaaa; border: 1px solid #333; border-radius: 4px; 
                        margin-top: 10px; padding-top: 15px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
            QPushButton { background-color: #2a2a2a; color: #ffffff; border: 1px solid #444; 
                          padding: 6px; border-radius: 3px; font-weight: bold; }
            QPushButton:hover { background-color: #3a3a3a; }
            QLineEdit, QComboBox { background-color: #2a2a2a; color: #ffffff; 
                                   border: 1px solid #444; padding: 4px; border-radius: 2px; }
            QCheckBox { color: #aaaaaa; }
            QLabel { color: #aaaaaa; }
        """)

    def toggle_simulation(self):
        if self.is_running:
            self.timer.stop()
            self.btn_start.setText("Start")
        else:
            self.timer.start()
            self.btn_start.setText("Pause")
        self.is_running = not self.is_running

    def save_history(self):
        self.history.append((self.ca_widget.age_grid.copy(), self.ca_widget.color_grid.copy()))

    def step_and_update(self):
        steps = 5 if self.turbo_mode else 1
        for _ in range(steps):
            self.save_history()
            self.ca_widget.step()
            self.generation += 1
            
        pop = self.ca_widget.count_alive()
        diversity = self.ca_widget.get_genetic_diversity()
        avg_color = self.ca_widget.get_average_color()
        
        self.lbl_generation.setText(f"Generation: {self.generation}")
        self.lbl_population.setText(f"Population: {pop}")
        self.lbl_diversity.setText(f"Diversity: {diversity:.1f}%")
        self.graph_widget.add_data(pop, diversity, avg_color)
        
        # Save for CSV
        self.csv_history.append({
            "gen": self.generation,
            "pop": pop,
            "diversity": diversity,
            "r": avg_color.red(),
            "g": avg_color.green(),
            "b": avg_color.blue()
        })

    def step_once(self):
        if not self.is_running:
            self.step_and_update()

    def rewind(self):
        if self.history:
            age, color = self.history.pop()
            self.ca_widget.age_grid = age
            self.ca_widget.color_grid = color
            self.generation = max(0, self.generation - 1)
            self.lbl_generation.setText(f"Generation: {self.generation}")
            self.lbl_population.setText(f"Population: {self.ca_widget.count_alive()}")
            self.ca_widget.update()

    def ca_clear(self):
        self.ca_widget.clear_grid()
        self.generation = 0
        self.history.clear()
        self.graph_widget.clear()
        self.csv_history.clear()
        self.lbl_generation.setText(f"Generation: {self.generation}")
        self.lbl_population.setText(f"Population: 0")
        self.lbl_diversity.setText(f"Diversity: 0.0%")

    def change_preset(self, text):
        if "HighLife" in text: b, s = "36", "23"
        elif "Conway" in text: b, s = "3", "23"
        elif "Fredkin" in text: b, s = "1357", "02468"
        elif "Seeds" in text: b, s = "2", ""
        else: return
        
        self.birth_input.setText(b)
        self.survive_input.setText(s)
        self.apply_rules()
        
        if "Fredkin" in text: self.change_pattern("Fredkin Cross")
        elif "HighLife" in text: self.change_pattern("HighLife Replicator")

    def change_pattern(self, text):
        self.ca_widget.current_pattern_name = text
        self.ca_widget.load_pattern(text)
        self.ca_clear()

    def toggle_stamp(self, state): self.ca_widget.stamp_mode = bool(state)
    def toggle_wrap(self, state): self.ca_widget.wrap_edges = bool(state)
    def toggle_vn(self, state): self.ca_widget.use_von_neumann = bool(state)
    def toggle_turbo(self, state): self.turbo_mode = bool(state)
    def toggle_grid(self, state): self.ca_widget.show_grid_lines = bool(state); self.ca_widget.update()
    def toggle_heatmap(self, state): 
        self.ca_widget.view_mode = "HEATMAP" if state else "GENETIC"
        self.ca_widget.update()
    
    def apply_rules(self):
        self.ca_widget.set_rules(self.birth_input.text(), self.survive_input.text())

    def update_speed(self): self.timer.setInterval(self.speed_slider.value())
    def update_zoom(self, size): self.ca_widget.update_cell_size(size)
    def update_brush(self, size): self.ca_widget.brush_size = size
    def update_mutation(self, val): self.ca_widget.mutation_rate = val / 1000.0
    def update_capacity(self, val): self.ca_widget.capacity = val / 100.0
        
    def choose_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.ca_widget.draw_color = np.array([color.red(), color.green(), color.blue()], dtype=np.uint8)
            self.btn_color.setStyleSheet(f"background-color: {color.name()}; color: black;")

    def save_image(self):
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Snapshot", "", "PNG Images (*.png)")
        if filepath:
            if not filepath.endswith('.png'): filepath += '.png'
            self.ca_widget.save_image(filepath)
            
    def export_csv(self):
        filepath, _ = QFileDialog.getSaveFileName(self, "Export CSV Data", "", "CSV Files (*.csv)")
        if filepath:
            if not filepath.endswith('.csv'): filepath += '.csv'
            with open(filepath, 'w') as f:
                f.write("Generation,Population,Diversity,Avg_R,Avg_G,Avg_B\n")
                for row in self.csv_history:
                    f.write(f"{row['gen']},{row['pop']},{row['diversity']:.2f},{row['r']},{row['g']},{row['b']}\n")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())