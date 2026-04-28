import random
from config import GRID_WIDTH, SHAPES, PIECE_TYPES

class Piece:
    def __init__(self, shape_type=None):
        if shape_type is None:
            self.type = random.choice(PIECE_TYPES)
        else:
            self.type = shape_type
            
        self.shapes = SHAPES[self.type]
        self.rotation_state = 0
        
        # 스폰 위치 (그리드 중앙 상단)
        self.x = GRID_WIDTH // 2 - len(self.get_shape()[0]) // 2
        self.y = 0
        
    def get_shape(self):
        """현재 회전 상태의 2차원 배열을 반환합니다."""
        return self.shapes[self.rotation_state % len(self.shapes)]
        
    def rotate(self, clockwise=True):
        """주어진 방향으로 회전 인덱스를 변경합니다. (엔진에서 충돌 시 되돌릴 수 있음)"""
        if clockwise:
            self.rotation_state = (self.rotation_state + 1) % len(self.shapes)
        else:
            self.rotation_state = (self.rotation_state - 1) % len(self.shapes)
