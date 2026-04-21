import pytest
import os
import sys

# 루트 경로 추가 (pytest 시 모듈 인식용)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import GRID_WIDTH, GRID_HEIGHT, DIR_RIGHT
from src.engine import TetrisEngine

def test_engine_initialization():
    engine = TetrisEngine()
    assert engine.score == 0
    assert engine.level == 1
    assert engine.lines_cleared == 0
    assert not engine.game_over
    assert engine.current_piece is not None
    # 초기 보드가 전부 0(빈칸)이어야 함
    assert all(cell == 0 for row in engine.board for cell in row)

def test_move_piece():
    engine = TetrisEngine()
    initial_x = engine.current_piece.x
    initial_y = engine.current_piece.y
    
    # 우측 이동 테스트
    moved = engine.move(DIR_RIGHT, 0)
    assert moved == True
    assert engine.current_piece.x == initial_x + 1
    assert engine.current_piece.y == initial_y
    
def test_hard_drop_and_lock():
    engine = TetrisEngine()
    piece_type = engine.current_piece.type
    
    # 하드 드롭으로 바닥까지 내림
    engine.hard_drop()
    
    # 보드 어딘가에 해당 조각 타입이 찍혀(문자열 저장) 있어야 함
    assert any(cell == piece_type for row in engine.board for cell in row)
    # 다음 조각이 스폰됨
    assert engine.current_piece is not None

def test_line_clear_logic():
    engine = TetrisEngine()
    
    # 마지막 한 줄(GRID_HEIGHT-1)을 강제로 채움
    engine.board[GRID_HEIGHT - 1] = ["TEST"] * GRID_WIDTH
    
    # 클리어 로직 호출
    engine.clear_lines()
    
    # 줄이 지워지고 level/score 갱신
    assert engine.lines_cleared == 1
    assert engine.score > 0
    
    # 줄이 지워진 후 바닥은 비어있어야 함(새 줄로 채워졌으므로)
    assert all(cell == 0 for cell in engine.board[GRID_HEIGHT - 1])
