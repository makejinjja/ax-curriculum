import pygame
from config import BLOCK_SIZE, COLORS, GRID_WIDTH, GRID_HEIGHT, SCREEN_WIDTH, SCREEN_HEIGHT

class Renderer:
    """좌/우 분할 화면을 지원하도록 리팩토링된 뷰"""
    def __init__(self):
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 20, bold=True)
        self.large_font = pygame.font.SysFont('Arial', 24, bold=True)
        self.vs_font = pygame.font.SysFont('Arial', 50, bold=True, italic=True)
        
    def draw_board(self, screen, board, base_x, base_y):
        board_rect = pygame.Rect(base_x, base_y, GRID_WIDTH * BLOCK_SIZE, GRID_HEIGHT * BLOCK_SIZE)
        pygame.draw.rect(screen, COLORS['BLACK'], board_rect)
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                rect = pygame.Rect(base_x + x * BLOCK_SIZE, base_y + y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
                pygame.draw.rect(screen, COLORS['GRID_COLOR'], rect, 1)
                
                if board[y][x] != 0:
                    self.draw_block(screen, rect, COLORS[board[y][x]])
                    
    def draw_piece(self, screen, piece, base_x, base_y, is_ghost=False):
        shape = piece.get_shape()
        color = COLORS[piece.type]
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    board_x = piece.x + x
                    board_y = piece.y + y
                    if 0 <= board_y < GRID_HEIGHT:
                        rect = pygame.Rect(
                            base_x + board_x * BLOCK_SIZE,
                            base_y + board_y * BLOCK_SIZE,
                            BLOCK_SIZE, BLOCK_SIZE
                        )
                        if is_ghost:
                            pygame.draw.rect(screen, color, rect, 2)
                        else:
                            self.draw_block(screen, rect, color)

    def draw_block(self, screen, rect, color, border_color=(255, 255, 255)):
        pygame.draw.rect(screen, color, rect)
        pygame.draw.rect(screen, border_color, rect, 1)
        inner_rect = rect.inflate(-6, -6)
        lighter = tuple(min(255, c + 60) for c in color)
        pygame.draw.rect(screen, lighter, inner_rect)
        
    def draw_mini_piece(self, screen, piece_type, x, y, inactive=False):
        from src.piece import Piece
        dummy = Piece(piece_type)
        shape = dummy.shapes[0]
        color = COLORS['GRAY'] if inactive else COLORS[piece_type]
        mini_size = BLOCK_SIZE - 10
        
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    rect = pygame.Rect(x + c * mini_size, y + r * mini_size, mini_size, mini_size)
                    self.draw_block(screen, rect, color, border_color=(100,100,100) if inactive else (255,255,255))

    def draw_engine_ui(self, screen, engine, base_x, base_y, participant_name):
        """주어진 베이스 X/Y를 기준으로 엔진과 UI 전체 세트를 그립니다."""
        self.draw_board(screen, engine.board, base_x, base_y)
        
        if engine.current_piece:
            ghost_y = engine.current_piece.y
            while not engine._check_collision(engine.current_piece, offset_y=(ghost_y - engine.current_piece.y + 1)):
                ghost_y += 1
            original_y = engine.current_piece.y
            engine.current_piece.y = ghost_y
            self.draw_piece(screen, engine.current_piece, base_x, base_y, is_ghost=True)
            engine.current_piece.y = original_y
            self.draw_piece(screen, engine.current_piece, base_x, base_y)

        # 왼쪽 UI (이름, Hold, 스코어)
        name_title = self.large_font.render(participant_name, True, COLORS['J'])
        screen.blit(name_title, (base_x - 120, base_y - 30))

        hold_title = self.font.render("HOLD", True, COLORS['WHITE'])
        screen.blit(hold_title, (base_x - 100, base_y + 30))
        if engine.hold_piece_type:
            self.draw_mini_piece(screen, engine.hold_piece_type, base_x - 100, base_y + 60, inactive=not engine.can_hold)

        score_t = self.font.render("SCORE", True, COLORS['WHITE'])
        score_v = self.large_font.render(str(engine.score), True, COLORS['I'])
        screen.blit(score_t, (base_x - 100, base_y + 180))
        screen.blit(score_v, (base_x - 100, base_y + 210))
        
        level_t = self.font.render("LEVEL", True, COLORS['WHITE'])
        level_v = self.large_font.render(str(engine.level), True, COLORS['O'])
        screen.blit(level_t, (base_x - 100, base_y + 260))
        screen.blit(level_v, (base_x - 100, base_y + 290))
        
        # 오른쪽 UI (Next)
        next_t = self.font.render("NEXT", True, COLORS['WHITE'])
        right_x = base_x + GRID_WIDTH * BLOCK_SIZE + 20
        screen.blit(next_t, (right_x, base_y + 30))
        
        if hasattr(engine, 'next_piece_types'):
            for i, p_type in enumerate(engine.next_piece_types[:3]):
                self.draw_mini_piece(screen, p_type, right_x, base_y + 70 + i * 80)

    def render(self, screen, player_engine, ai_engine):
        screen.fill(COLORS['GRAY'])
        
        # VS 마크 중앙 하단 부근에 렌더링
        vs_text = self.vs_font.render("VS", True, COLORS['WHITE'])
        screen.blit(vs_text, (SCREEN_WIDTH // 2 - vs_text.get_width() // 2, SCREEN_HEIGHT // 2))

        # 플레이어 1: 좌측 (기준 X: 150)
        self.draw_engine_ui(screen, player_engine, 150, 50, "P1 PLAYER")
        
        # AI 경쟁자: 우측 (기준 X: 650)
        self.draw_engine_ui(screen, ai_engine, 650, 50, "CPU BOT")
