import copy
from config import GRID_WIDTH, GRID_HEIGHT

class TetrisAI:
    def __init__(self, engine):
        self.engine = engine
        self.action_queue = []
        self.last_action_time = 0

    def calculate_best_move(self):
        """가장 점수가 높은 X위치와 회전 상태에 대한 커맨드 시퀀스를 반환"""
        best_score = -float('inf')
        best_x = 0
        best_r = 0
        
        if not self.engine.current_piece: 
            return []
            
        original_piece = copy.deepcopy(self.engine.current_piece)
        original_board = [row[:] for row in self.engine.board]
        
        for r in range(len(original_piece.shapes)):
            test_piece = copy.deepcopy(original_piece)
            test_piece.rotation_state = r
            
            for x in range(-2, GRID_WIDTH + 2):
                test_piece.x = x
                test_piece.y = 0
                
                if self.engine._check_collision(test_piece):
                    continue
                
                # 강제 낙하 위치 확인
                ghost_y = test_piece.y
                while not self.engine._check_collision(test_piece, offset_y=(ghost_y - test_piece.y + 1)):
                    ghost_y += 1
                test_piece.y = ghost_y
                
                # 가상 보드 록다운 생성
                sim_board = [row[:] for row in original_board]
                for p_y, row in enumerate(test_piece.get_shape()):
                    for p_x, cell in enumerate(row):
                        if cell:
                            if 0 <= test_piece.y + p_y < GRID_HEIGHT:
                                sim_board[test_piece.y + p_y][test_piece.x + p_x] = test_piece.type
                
                score = self.evaluate_board(sim_board)
                if score > best_score:
                    best_score = score
                    best_x = x
                    best_r = r
                    
        # 액션 시퀀스 생성
        curr_r = original_piece.rotation_state % len(original_piece.shapes)
        diff_r = (best_r - curr_r) % len(original_piece.shapes)
        
        actions = ['ROTATE'] * diff_r
        diff_x = best_x - original_piece.x
        if diff_x < 0:
            actions += ['LEFT'] * abs(diff_x)
        else:
            actions += ['RIGHT'] * diff_x
            
        actions.append('HARD_DROP')
        return actions

    def evaluate_board(self, board):
        """휴리스틱으로 보드를 평가합니다."""
        heights = [0] * GRID_WIDTH
        holes = 0
        complete_lines = 0
        
        for y in range(GRID_HEIGHT):
            is_complete = True
            for x in range(GRID_WIDTH):
                if board[y][x] != 0:
                    if heights[x] == 0:
                        heights[x] = GRID_HEIGHT - y
                else:
                    is_complete = False
                    if heights[x] > 0:
                        holes += 1
            if is_complete:
                complete_lines += 1
                
        aggregate_height = sum(heights)
        bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(GRID_WIDTH-1))
        
        # AI Heuristic Weights
        return (-0.510066 * aggregate_height) + (0.760666 * complete_lines) + (-0.35663 * holes) + (-0.184483 * bumpiness)

    def update(self, dt):
        """주기적으로 액션 큐를 비워가며 입력명령 발생"""
        if self.engine.game_over: return
        
        # 레벨이 높을수록 딜레이 감축 (초기 레벨엔 0.5초마다 키를 눌러 둔하다가 레벨 업하면 기계 같아짐)
        delay = max(20, 500 - (self.engine.level - 1) * 35)
        self.last_action_time += dt
        
        if not self.action_queue:
            self.action_queue = self.calculate_best_move()
            # 초반에는 생각하는 시간을 위해 조금 더 딜레이 누적 보너스 부여
            self.last_action_time = -100
            
        if self.action_queue and self.last_action_time >= delay:
            action = self.action_queue.pop(0)
            if action == 'ROTATE': self.engine.rotate()
            elif action == 'LEFT': self.engine.move(-1, 0)
            elif action == 'RIGHT': self.engine.move(1, 0)
            elif action == 'HARD_DROP': self.engine.hard_drop()
            self.last_action_time = 0
