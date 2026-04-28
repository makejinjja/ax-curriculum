import pygame
from config import SCREEN_WIDTH, SCREEN_HEIGHT
from src.states import StateManager

def main():
    # Windows DPI 인식을 통해 흐릿함을 방지
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Tetris - Vibe Coding")
    
    manager = StateManager(screen)
    manager.run()

if __name__ == "__main__":
    main()
