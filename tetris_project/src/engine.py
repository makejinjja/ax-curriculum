from config import GRID_WIDTH, GRID_HEIGHT, DIR_LEFT, DIR_RIGHT, DIR_DOWN
from src.piece import Piece

class TetrisEngine:
    """테트리스 논리 흐름을 관장하는 모듈 (Pygame 완전 독립)"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        # 0: 빈칸, 문자열: 블록 타일 색상 기준
        self.board = [[0] * GRID_WIDTH for _ in range(GRID_HEIGHT)]
        self.current_piece = None
        self.hold_piece_type = None
        self.can_hold = True
        
        from src.piece import Piece
        # 미리 3개의 넥스트 블록 채우기
        self.next_piece_types = [Piece().type for _ in range(3)]
        
        self.score = 0
        self.level = 1
        self.lines_cleared = 0
        self.game_over = False
        self.outgoing_attack = 0
        
        self.spawn_piece()
        
    def spawn_piece(self, piece_type=None):
        """보드 상단에 새 블록을 내보냅니다."""
        from src.piece import Piece
        if piece_type:
            self.current_piece = Piece(piece_type)
        else:
            self.current_piece = Piece(self.next_piece_types.pop(0))
            self.next_piece_types.append(Piece().type)
            
        self.can_hold = True
        
        if self._check_collision(self.current_piece):
            self.game_over = True
            
    def _check_collision(self, piece, offset_x=0, offset_y=0):
        """piece가 보드 경계나 바닥, 다른 블록에 충돌하는지 검사합니다."""
        shape = piece.get_shape()
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    new_x = piece.x + x + offset_x
                    new_y = piece.y + y + offset_y
                    
                    if new_x < 0 or new_x >= GRID_WIDTH or new_y >= GRID_HEIGHT:
                        return True
                    if new_y >= 0 and self.board[new_y][new_x] != 0:
                        return True
        return False
        
    def move(self, dx, dy):
        """지정된 오프셋으로 블록을 움직입니다."""
        if not self._check_collision(self.current_piece, offset_x=dx, offset_y=dy):
            self.current_piece.x += dx
            self.current_piece.y += dy
            return True
        return False
        
    def rotate(self, clockwise=True):
        """현재 블록을 회전시킵니다. 충돌 발생 시 롤백합니다."""
        self.current_piece.rotate(clockwise)
        if self._check_collision(self.current_piece):
            self.current_piece.rotate(not clockwise)
            return False
        return True
        
    def lock_piece(self):
        """블록이 바닥에 닿았을 때 보드에 고정하고 점수 판정을 내립니다."""
        shape = self.current_piece.get_shape()
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    board_y = self.current_piece.y + y
                    board_x = self.current_piece.x + x
                    if 0 <= board_y < GRID_HEIGHT:
                        self.board[board_y][board_x] = self.current_piece.type
                        
        self.clear_lines()
        self.spawn_piece()
        
    def clear_lines(self):
        """꽉 찬 줄을 삭제하고 점수를 올립니다."""
        lines_to_clear = []
        for y in range(GRID_HEIGHT):
            if all(cell != 0 for cell in self.board[y]):
                lines_to_clear.append(y)
                
        if not lines_to_clear:
            return
            
        num_lines = len(lines_to_clear)
        
        # 지워지지 않은 줄 위에 빈 줄 채워넣기
        new_board = [[0] * GRID_WIDTH for _ in range(num_lines)]
        for y in range(GRID_HEIGHT):
            if y not in lines_to_clear:
                new_board.append(self.board[y])
        self.board = new_board
        
        self.lines_cleared += num_lines
        self.score += [0, 40, 100, 300, 1200][num_lines] * self.level
        self.level = self.lines_cleared // 10 + 1
        
        # 멀티플레이어 공격 포인트 연산 (테트리스 룰)
        attack_table = {1: 0, 2: 1, 3: 2, 4: 4}
        if num_lines in attack_table:
            self.outgoing_attack += attack_table[num_lines]
        
    def drop(self):
        """강제 중력 스텝: 떨어질 수 없다면 조각을 보드에 록다운합니다."""
        if self.game_over:
            return False
        if self.move(0, DIR_DOWN):
            return True
        self.lock_piece()
        return False
        
    def hard_drop(self):
        """하드 드롭: 닿을 때까지 빠르게 내리고 즉시 고정합니다."""
        if self.game_over: return
        while self.move(0, DIR_DOWN):
            pass
        self.lock_piece()

    def hold(self):
        """현재 큐에 있는 조각을 킵(Hold)합니다. (턴당 1회 제한)"""
        if self.game_over or not self.can_hold:
            return
            
        # 기존 홀드 피스가 없으면 새 피스 pop, 있으면 스왑
        if self.hold_piece_type is None:
            self.hold_piece_type = self.current_piece.type
            self.spawn_piece()
        else:
            temp = self.hold_piece_type
            self.hold_piece_type = self.current_piece.type
            self.spawn_piece(temp)
            
        self.can_hold = False

    def add_garbage_lines(self, amount):
        """상대방의 공격으로 하단 가비지 라인이 쌓입니다."""
        if amount <= 0 or self.game_over:
            return
            
        import random
        # 상단 amount 개수만큼의 윗줄에 블록이 있었다면 화면 밖으로 밀려나므로 게임오버 판정
        for y in range(amount):
            if any(cell != 0 for cell in self.board[y]):
                self.game_over = True
                
        # 리스트 슬라이싱으로 전체 라인을 위로 amount칸 밀어올림
        self.board = self.board[amount:]
        
        # 하단에 구멍 뚫린 가비지 라인 추가
        for _ in range(amount):
            hole_idx = random.randint(0, GRID_WIDTH - 1)
            new_row = ['GRAY'] * GRID_WIDTH
            new_row[hole_idx] = 0
            self.board.append(new_row)
            
        # 떨어지고 있던 현재 조각도 겹치지 않게 같이 밀어올림
        if self.current_piece:
            self.current_piece.y -= amount
