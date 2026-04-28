import pygame
import sys
from config import SCREEN_WIDTH, SCREEN_HEIGHT, COLORS, DIR_LEFT, DIR_RIGHT
from src.engine import TetrisEngine
from src.renderer import Renderer

class State:
    def __init__(self, state_manager):
        self.state_manager = state_manager
    def handle_events(self, events): pass
    def update(self, dt): pass
    def render(self, screen): pass

class MenuState(State):
    def __init__(self, state_manager):
        super().__init__(state_manager)
        pygame.font.init()
        self.title_font = pygame.font.SysFont('Arial', 64, bold=True)
        self.font = pygame.font.SysFont('Arial', 24)

    def handle_events(self, events):
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # 새로 게임 인스턴스를 만들어 시작
                    self.state_manager.states['playing'] = PlayingState(self.state_manager)
                    self.state_manager.change_state('playing')

    def render(self, screen):
        screen.fill(COLORS['BLACK'])
        title = self.title_font.render("1vs1 AI BATTLE", True, COLORS['J'])
        prompt = self.font.render("Press SPACE to Start", True, COLORS['WHITE'])
        screen.blit(title, (SCREEN_WIDTH//2 - title.get_width()//2, SCREEN_HEIGHT//3))
        screen.blit(prompt, (SCREEN_WIDTH//2 - prompt.get_width()//2, SCREEN_HEIGHT//2))

class PauseState(State):
    def __init__(self, state_manager):
        super().__init__(state_manager)
        self.font = pygame.font.SysFont('Arial', 48, bold=True)

    def handle_events(self, events):
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_p, pygame.K_SPACE):
                    self.state_manager.change_state('playing')

    def render(self, screen):
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(150)
        overlay.fill((0, 0, 0))
        screen.blit(overlay, (0, 0))
        
        text = self.font.render("PAUSED", True, COLORS['WHITE'])
        screen.blit(text, (SCREEN_WIDTH//2 - text.get_width()//2, SCREEN_HEIGHT//2 - 30))

class GameOverState(State):
    def __init__(self, state_manager):
        super().__init__(state_manager)
        self.font = pygame.font.SysFont('Arial', 48, bold=True)
        self.small = pygame.font.SysFont('Arial', 24)
        self.winner_text = ""

    def handle_events(self, events):
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.state_manager.states['playing'] = PlayingState(self.state_manager)
                    self.state_manager.change_state('playing')

    def render(self, screen):
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(150)
        overlay.fill((0, 0, 0))
        screen.blit(overlay, (0, 0))
        text = self.font.render("GAME OVER", True, COLORS['Z'])
        w_text = self.font.render(self.winner_text, True, COLORS['T'])
        prompt = self.small.render("Press SPACE to Restart", True, COLORS['WHITE'])
        screen.blit(text, (SCREEN_WIDTH//2 - text.get_width()//2, SCREEN_HEIGHT//3 - 30))
        screen.blit(w_text, (SCREEN_WIDTH//2 - w_text.get_width()//2, SCREEN_HEIGHT//3 + 40))
        screen.blit(prompt, (SCREEN_WIDTH//2 - prompt.get_width()//2, SCREEN_HEIGHT//2 + 30))

class PlayingState(State):
    def __init__(self, state_manager):
        super().__init__(state_manager)
        self.player_engine = TetrisEngine()
        self.ai_engine = TetrisEngine()
        
        from src.ai import TetrisAI
        self.ai = TetrisAI(self.ai_engine)
        self.renderer = Renderer()
        
        self.fall_time_p = 0
        self.fall_time_ai = 0

    def handle_events(self, events):
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_p:
                    self.state_manager.change_state('paused')
                elif not self.player_engine.game_over:
                    if event.key == pygame.K_LEFT:
                        self.player_engine.move(DIR_LEFT, 0)
                    elif event.key == pygame.K_RIGHT:
                        self.player_engine.move(DIR_RIGHT, 0)
                    elif event.key == pygame.K_DOWN:
                        self.player_engine.move(0, 1)
                        self.player_engine.score += 2 # 소프트 드롭
                    elif event.key == pygame.K_UP:
                        self.player_engine.rotate()
                    elif event.key == pygame.K_SPACE:
                        self.player_engine.hard_drop()
                    elif event.key in [pygame.K_c, pygame.K_LSHIFT]:
                        self.player_engine.hold()

    def update(self, dt):
        if self.player_engine.game_over or self.ai_engine.game_over:
            if self.player_engine.game_over and not self.ai_engine.game_over:
                self.state_manager.states['gameover'].winner_text = "AI BOT WINS!"
            elif self.ai_engine.game_over and not self.player_engine.game_over:
                self.state_manager.states['gameover'].winner_text = "PLAYER 1 WINS!"
            else:
                self.state_manager.states['gameover'].winner_text = "DRAW!"
                
            self.state_manager.change_state('gameover')
            return
            
        self.fall_time_p += dt
        self.fall_time_ai += dt
        
        speed_p = max(50, 600 - (self.player_engine.level - 1) * 60)
        speed_ai = max(50, 600 - (self.ai_engine.level - 1) * 60)
        
        if self.fall_time_p >= speed_p:
            self.player_engine.drop()
            self.fall_time_p = 0
            
        if self.fall_time_ai >= speed_ai:
            self.ai_engine.drop()
            self.fall_time_ai = 0
            
        self.ai.update(dt)
        
        # 교차 공격 적용 (쓰레기 줄 발송)
        if self.player_engine.outgoing_attack > 0:
            self.ai_engine.add_garbage_lines(self.player_engine.outgoing_attack)
            self.player_engine.outgoing_attack = 0
            
        if self.ai_engine.outgoing_attack > 0:
            self.player_engine.add_garbage_lines(self.ai_engine.outgoing_attack)
            self.ai_engine.outgoing_attack = 0

    def render(self, screen):
        self.renderer.render(screen, self.player_engine, self.ai_engine)

class StateManager:
    def __init__(self, screen):
        self.screen = screen
        self.states = {
            'menu': MenuState(self),
            'playing': PlayingState(self),
            'paused': PauseState(self),
            'gameover': GameOverState(self)
        }
        self.current_state_name = 'menu'
        self.current_state = self.states[self.current_state_name]

    def change_state(self, name):
        self.current_state_name = name
        self.current_state = self.states[name]

    def run(self):
        clock = pygame.time.Clock()
        from config import FPS
        while True:
            dt = clock.tick(FPS)
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                    
            self.current_state.handle_events(events)
            self.current_state.update(dt)
            
            if self.current_state_name in ['paused', 'gameover']:
                self.states['playing'].render(self.screen)
                self.current_state.render(self.screen)
            else:
                self.current_state.render(self.screen)
                
            pygame.display.flip()
