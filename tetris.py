import tkinter as tk
import random
import os

class Tetris(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.cell_size = 30
        self.cols = 10
        self.rows = 20
        
        # 테트리스 블록 (테트로미노) 정의
        self.shapes = [
            [[1, 1, 1, 1]], # I, cyan
            [[1, 1], [1, 1]], # O, yellow
            [[0, 1, 0], [1, 1, 1]], # T, purple
            [[1, 0, 0], [1, 1, 1]], # L, orange
            [[0, 0, 1], [1, 1, 1]], # J, blue
            [[0, 1, 1], [1, 1, 0]], # S, green
            [[1, 1, 0], [0, 1, 1]]  # Z, red
        ]
        self.colors = ['cyan', 'yellow', 'purple', 'orange', 'blue', 'green', 'red']
        
        self.board = [[0] * self.cols for _ in range(self.rows)]
        self.score = 0
        self.high_score = self.load_high_score()
        self.game_over = False
        
        # Piece states
        self.current_piece = None
        self.current_color = None
        self.current_piece_idx = None
        self.piece_x = 0
        self.piece_y = 0

        self.next_piece_idx = random.randint(0, len(self.shapes) - 1)
        self.hold_piece_idx = None
        self.can_hold = True
        
        self.setup_ui()
        self.master.bind("<Key>", self.handle_events)
        
        self.spawn_piece()
        self.update_clock()

    def setup_ui(self):
        # 전체 레이아웃 (좌측 보드, 우측 정보패널)
        self.pack(padx=20, pady=20)
        
        # 메인 테트리스 보드 (Canvas)
        self.board_canvas = tk.Canvas(self, width=self.cols * self.cell_size, 
                                      height=self.rows * self.cell_size, bg='#1a1a1a', highlightthickness=1)
        self.board_canvas.grid(row=0, column=0, rowspan=5, padx=(0, 20))
        
        font_large = ("Helvetica", 14, "bold")
        font_small = ("Helvetica", 10, "bold")
        
        # 점수표 (Scoreboard) 패널
        self.score_frame = tk.Frame(self)
        self.score_frame.grid(row=0, column=1, sticky="n")
        
        tk.Label(self.score_frame, text="SCORE", font=font_small, fg="gray").pack(pady=(0, 2))
        self.score_label = tk.Label(self.score_frame, text=str(self.score), font=font_large)
        self.score_label.pack(pady=(0, 15))
        
        tk.Label(self.score_frame, text="HIGH SCORE", font=font_small, fg="gray").pack(pady=(0, 2))
        self.high_score_label = tk.Label(self.score_frame, text=str(self.high_score), font=font_large)
        self.high_score_label.pack()
        
        # 다음 블록 (NEXT) 패널
        tk.Label(self, text="NEXT", font=font_small, fg="gray").grid(row=1, column=1, sticky="s", pady=(20, 5))
        self.next_canvas = tk.Canvas(self, width=4 * self.cell_size, height=4 * self.cell_size, bg='#262626', highlightthickness=0)
        self.next_canvas.grid(row=2, column=1, pady=(0, 20))
        
        # 홀드 블록 (HOLD) 패널
        tk.Label(self, text="HOLD (Press 'C')", font=font_small, fg="gray").grid(row=3, column=1, sticky="s", pady=(0, 5))
        self.hold_canvas = tk.Canvas(self, width=4 * self.cell_size, height=4 * self.cell_size, bg='#262626', highlightthickness=0)
        self.hold_canvas.grid(row=4, column=1, sticky="n")

    def load_high_score(self):
        if os.path.exists('high_score.txt'):
            try:
                with open('high_score.txt', 'r') as f:
                    return int(f.read().strip())
            except:
                return 0
        return 0

    def save_high_score(self):
        if self.score > self.high_score:
            self.high_score = self.score
            with open('high_score.txt', 'w') as f:
                f.write(str(self.high_score))

    def update_score(self, points):
        self.score += points
        self.score_label.config(text=str(self.score))
        if self.score > self.high_score:
            self.high_score_label.config(text=str(self.score))

    def draw_mini_piece(self, canvas, piece_idx):
        canvas.delete("all")
        if piece_idx is None:
            return
        piece = self.shapes[piece_idx]
        color = self.colors[piece_idx]
        
        piece_height = len(piece)
        piece_width = len(piece[0])
        
        # 4x4 캔버스의 중심에 오도록 계산
        start_x = (4 * self.cell_size - piece_width * self.cell_size) / 2
        start_y = (4 * self.cell_size - piece_height * self.cell_size) / 2
        
        for r, row in enumerate(piece):
            for c, val in enumerate(row):
                if val:
                    canvas.create_rectangle(start_x + c * self.cell_size, start_y + r * self.cell_size,
                                            start_x + (c + 1) * self.cell_size, start_y + (r + 1) * self.cell_size,
                                            fill=color, outline="black")

    def spawn_piece(self):
        self.current_piece_idx = self.next_piece_idx
        self.current_piece = self.shapes[self.current_piece_idx]
        self.current_color = self.colors[self.current_piece_idx]
        
        # 새로운 다음 블록 설정 및 그리기
        self.next_piece_idx = random.randint(0, len(self.shapes) - 1)
        self.draw_mini_piece(self.next_canvas, self.next_piece_idx)
        
        self.can_hold = True # 블록이 스폰될 때마다 홀드 기능 활성화
        
        self.piece_x = self.cols // 2 - len(self.current_piece[0]) // 2
        self.piece_y = 0
        
        if self.check_collision(self.current_piece, self.piece_x, self.piece_y):
            self.game_over = True
            self.save_high_score()

    def hold_piece(self):
        if not self.can_hold:
            return
            
        if self.hold_piece_idx is None:
            self.hold_piece_idx = self.current_piece_idx
            self.spawn_piece()
        else:
            # 현재 블록과 홀드 블록을 스왑
            self.current_piece_idx, self.hold_piece_idx = self.hold_piece_idx, self.current_piece_idx
            self.current_piece = self.shapes[self.current_piece_idx]
            self.current_color = self.colors[self.current_piece_idx]
            
            self.piece_x = self.cols // 2 - len(self.current_piece[0]) // 2
            self.piece_y = 0
            
        # 홀드 캔버스 업데이트
        self.draw_mini_piece(self.hold_canvas, self.hold_piece_idx)
        self.can_hold = False # 다시 떨어질 때까지 홀드 불가
        
        # 보드 다시 그리기
        self.draw()

    def check_collision(self, piece, x, y):
        for r, row in enumerate(piece):
            for c, val in enumerate(row):
                if val:
                    if x + c < 0 or x + c >= self.cols or y + r >= self.rows or self.board[y + r][x + c]:
                        return True
        return False

    def merge_piece(self):
        for r, row in enumerate(self.current_piece):
            for c, val in enumerate(row):
                if val:
                    self.board[self.piece_y + r][self.piece_x + c] = self.current_color
        self.clear_lines()
        self.spawn_piece()

    def clear_lines(self):
        lines_to_clear = [i for i, row in enumerate(self.board) if all(row)]
        for i in lines_to_clear:
            del self.board[i]
            self.board.insert(0, [0] * self.cols)
            
        if lines_to_clear:
            cleared = len(lines_to_clear)
            # 클리어한 줄의 개수에 따라 추가 점수
            points = cleared * 100 * cleared 
            self.update_score(points)

    def rotate_piece(self):
        new_piece = [list(row) for row in zip(*self.current_piece[::-1])]
        if not self.check_collision(new_piece, self.piece_x, self.piece_y):
            self.current_piece = new_piece

    def handle_events(self, event):
        if self.game_over:
            return
            
        if event.keysym == 'Left':
            if not self.check_collision(self.current_piece, self.piece_x - 1, self.piece_y):
                self.piece_x -= 1
        elif event.keysym == 'Right':
            if not self.check_collision(self.current_piece, self.piece_x + 1, self.piece_y):
                self.piece_x += 1
        elif event.keysym == 'Down':
            if not self.check_collision(self.current_piece, self.piece_x, self.piece_y + 1):
                self.piece_y += 1
                self.update_score(1) # Soft drop score
        elif event.keysym == 'Up':
            self.rotate_piece()
        elif event.keysym == 'space':
            # Hard drop
            drop_dist = 0
            while not self.check_collision(self.current_piece, self.piece_x, self.piece_y + 1):
                self.piece_y += 1
                drop_dist += 1
            self.update_score(drop_dist * 2) # Hard drop score
            self.merge_piece()
        elif event.keysym.lower() == 'c':
            self.hold_piece()
            
        self.draw()

    def update_clock(self):
        if not self.game_over:
            if not self.check_collision(self.current_piece, self.piece_x, self.piece_y + 1):
                self.piece_y += 1
            else:
                self.merge_piece()
            self.draw()
            self.master.after(500, self.update_clock)
        else:
            self.board_canvas.create_text(self.cols * self.cell_size / 2, self.rows * self.cell_size / 2,
                                          text="GAME OVER", fill="white", font=("Arial", 28, "bold"))
            self.save_high_score()

    def draw(self):
        self.board_canvas.delete("all")
        
        # 보드 그리기
        for r, row in enumerate(self.board):
            for c, color in enumerate(row):
                if color:
                    self.board_canvas.create_rectangle(c * self.cell_size, r * self.cell_size,
                                                       (c + 1) * self.cell_size, (r + 1) * self.cell_size,
                                                       fill=color, outline="#333333")
        
        # 낙하 예상 위치 (Ghost Piece) 보너스 기능
        ghost_y = self.piece_y
        while not self.check_collision(self.current_piece, self.piece_x, ghost_y + 1):
            ghost_y += 1
            
        for r, row in enumerate(self.current_piece):
            for c, val in enumerate(row):
                if val:
                    self.board_canvas.create_rectangle((self.piece_x + c) * self.cell_size,
                                                       (ghost_y + r) * self.cell_size,
                                                       (self.piece_x + c + 1) * self.cell_size,
                                                       (ghost_y + r + 1) * self.cell_size,
                                                       fill="", outline=self.current_color, dash=(2, 2))
        
        # 현재 떨어지는 블록 그리기
        for r, row in enumerate(self.current_piece):
            for c, val in enumerate(row):
                if val:
                    self.board_canvas.create_rectangle((self.piece_x + c) * self.cell_size,
                                                       (self.piece_y + r) * self.cell_size,
                                                       (self.piece_x + c + 1) * self.cell_size,
                                                       (self.piece_y + r + 1) * self.cell_size,
                                                       fill=self.current_color, outline="white")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Tetris")
    root.resizable(False, False)
    root.configure(bg='#f0f0f0')
    game = Tetris(root)
    root.mainloop()
