
import numpy as np
import pygame
from   pygame import K_ESCAPE, SCALED, mixer
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import os

pygame.init()
pygame.display.set_caption("Ludo AI")
screen = pygame.display.set_mode((600, 600),SCALED)

board = pygame.image.load('pics/board/Board1.jpg')
star  = pygame.image.load('pics/star.png')
one   = pygame.image.load('pics/dice/1.png')
two   = pygame.image.load('pics/dice/2.png')
three = pygame.image.load('pics/dice/3.png')
four  = pygame.image.load('pics/dice/4.png')
five  = pygame.image.load('pics/dice/5.png')
six   = pygame.image.load('pics/dice/6.png') 

red    = pygame.image.load('pics/gotiyan/red.png')
blue   = pygame.image.load('pics/gotiyan/blue.png')
green  = pygame.image.load('pics/gotiyan/green.png')
yellow = pygame.image.load('pics/gotiyan/yellow.png')

DICE  = [one, two, three, four, five, six]
color = [red, green, yellow, blue]

killSound   = mixer.Sound("sounds/Killed.wav")
tokenSound  = mixer.Sound("sounds/Token Movement.wav")
diceSound   = mixer.Sound("sounds/Dice Roll.wav")
winnerSound = mixer.Sound("sounds/Reached Star.wav")

number        = 1
currentPlayer = 0
playerKilled  = False
diceRolled    = False
winnerRank    = []
start = False

font = pygame.font.Font('freesansbold.ttf', 11)
FONT = pygame.font.Font('freesansbold.ttf', 16)


HOME = [[(110, 58),  (61, 107),  (152, 107), (110, 152)],  # Red
        [(466, 58),  (418, 107), (509, 107), (466, 153)],  # Green
        [(466, 415), (418, 464), (509, 464), (466, 510)],  # Yellow
        [(110, 415), (61, 464),  (152, 464), (110, 510)]]  # Blue

        # R          G          Y          B
SAFE = [(50, 240), (328, 50), (520, 328), (240, 520),
        (88, 328), (240, 88), (482, 240), (328, 482)]

DicePosition=[(175,173),(531,173),(531,375),(173,375)]

position = [[[110, 58],  [61, 107],  [152, 107], [110, 152]],  # Red
            [[466, 58],  [418, 107], [509, 107], [466, 153]],  # Green
            [[466, 415], [418, 464], [509, 464], [466, 510]],  # Yellow
            [[110, 415], [61, 464],  [152, 464], [110, 510]]]  # Blue

jump = {(202, 240): (240, 202),  # Red to Green
        (328, 202): (368, 240),  # Gren to yellow
        (368, 328): (328, 368),  # Yellow to blue
        (240, 368): (202, 328)}  # Blue to red

         # R           G            Y          B
WINNER = [[240, 284], [284, 240], [330, 284], [284, 330]]

winner_path = [
    
        [(12,284),(50, 284), (88, 284), (126, 284), (164, 284), (202, 284), (240, 284)],
        [(284,12),(284, 50), (284, 88), (284, 126), (284, 164), (284, 202), (284, 240)],
        [(558,284),(520, 284), (482, 284), (444, 284), (406, 284), (368, 284), (330, 284)],
        [(284,558),(284, 520), (284, 482), (284, 444), (284, 406), (284, 368), (284, 330)]      
    ]



pygame.freetype.get_default_font() 

def re_initialize():
    global number,currentPlayer,playerKilled,diceRolled,winnerRank,HOME,SAFE,DicePosition,position,jump,WINNER
    number   = 1
    currentPlayer = 0
    playerKilled  = False
    diceRolled    = False
    winnerRank    = []
        
    HOME = [[(110, 58),  (61, 107),  (152, 107), (110, 152)],  # Red
            [(466, 58),  (418, 107), (509, 107), (466, 153)],  # Green
            [(466, 415), (418, 464), (509, 464), (466, 510)],  # Yellow
            [(110, 415), (61, 464),  (152, 464), (110, 510)]]  # Blue

            # R          G          Y          B
    SAFE = [(50, 240), (328, 50), (520, 328), (240, 520),
            (88, 328), (240, 88), (482, 240), (328, 482)]
    DicePosition=[(175,173),(531,173),(531,375),(173,375)]
    position = [[[110, 58],  [61, 107],  [152, 107], [110, 152]],  # Red
                [[466, 58],  [418, 107], [509, 107], [466, 153]],  # Green
                [[466, 415], [418, 464], [509, 464], [466, 510]],  # Yellow
                [[110, 415], [61, 464],  [152, 464], [110, 510]]]  # Blue

    jump = {(202, 240): (240, 202),  # Red to Green
            (328, 202): (368, 240),  # Gren to yellow
            (368, 328): (328, 368),  # Yellow to blue
            (240, 368): (202, 328)}  # Blue to red

            # R           G            Y          B
    WINNER = [[240, 284], [284, 240], [330, 284], [284, 330]]
    
   
    
    
AI_PLAYER_INDEX_R = 0  # AI player 1
AI_PLAYER_INDEX_G = 1  # AI player 2
AI_PLAYER_INDEX_Y = 2  # AI player 3
AI_PLAYER_INDEX_B = 3  # AI player 4
NUM_ACTIONS = 4  


class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(32, 64)  # Adjusted input size
        self.fc2 = nn.Linear(64, NUM_ACTIONS)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

r_q_network = QNetwork()
g_q_network = QNetwork()
y_q_network = QNetwork()
b_q_network = QNetwork()
q_network = QNetwork()

r_model_path = 'paths/r_ludo_q_network.pth'
g_model_path = 'paths/g_ludo_q_network.pth'
y_model_path = 'paths/y_ludo_q_network.pth'
b_model_path = 'paths/b_ludo_q_network.pth'
model_path = 'paths/ludo_q_network.pth'

if os.path.exists(r_model_path):
    r_q_network.load_state_dict(torch.load(r_model_path))

if os.path.exists(g_model_path):
    g_q_network.load_state_dict(torch.load(g_model_path))

if os.path.exists(y_model_path):
    y_q_network.load_state_dict(torch.load(y_model_path))

if os.path.exists(b_model_path):
    b_q_network.load_state_dict(torch.load(b_model_path))

if os.path.exists(model_path):
    q_network.load_state_dict(torch.load(model_path))


r_optimizer = optim.Adam(r_q_network.parameters(), lr=0.001)
g_optimizer = optim.Adam(g_q_network.parameters(), lr=0.001)
y_optimizer = optim.Adam(y_q_network.parameters(), lr=0.001)
b_optimizer = optim.Adam(b_q_network.parameters(), lr=0.001)
optimizer = optim.Adam(q_network.parameters(), lr=0.001)

criterion = nn.MSELoss()

def preprocess_state(state):
    positions = state['positions']
    flat_positions = [pos for player in positions for token in player for pos in token]
    return flat_positions

def choose_action(state, q_network, epsilon=0.1):
    if random.random() < epsilon:
        return random.choice([0, 1, 2, 3])
    else:
        with torch.no_grad():

            state_tensor = torch.tensor(state, dtype=torch.float32)
            q_values = q_network(state_tensor)
            return torch.argmax(q_values).item()

def all_in_winner_rank(player):
    return all(pos in WINNER[player] for pos in position[player])

def token_reached_winner_rank(player, token_index):
    return position[player][token_index] in WINNER[player]

def token_on_winner_path(player, token_index):
    return position[player][token_index] in winner_path[player]

def token_reached_safe_spot(player, token_index):
    return position[player][token_index] in SAFE

def token_left_home(player, token_index):
    return position[player][token_index] not in HOME[player]

def token_moved_forward(old_state, new_state, player, token_index):
    old_pos = old_state['positions'][player][token_index]
    new_pos = new_state['positions'][player][token_index]
    return new_pos > old_pos  

def calculate_reward(old_state, new_state, action_taken, playerKilled, winnerRank, old_winner_rank):
    reward = 0
    if all_in_winner_rank(currentPlayer):
        reward += 100

    elif token_reached_winner_rank(currentPlayer, action_taken):
        reward += 50

    if token_on_winner_path(currentPlayer, action_taken):
        reward += 25

    if playerKilled:
        reward += 20

    if token_reached_safe_spot(currentPlayer, action_taken):
        reward += 10

    if token_left_home(currentPlayer, action_taken):
        reward += 5

    if token_moved_forward(old_state, new_state, currentPlayer, action_taken):
        reward += 1

    if old_state == new_state:
        reward -= 100

    return reward


def get_current_state():
    return [currentPlayer, number, len(position[currentPlayer])]  


def show_token(x, y):
    screen.fill((0, 0, 0))
    screen.blit(board, (0, 0))
    for i in SAFE[4:]:
        screen.blit(star, i)

    for i in range(len(position)):
        for j in position[i]:
            screen.blit(color[i], j)

    screen.blit(DICE[number-1], DicePosition[currentPlayer])

    if position[x][y] in WINNER:
        winnerSound.play()
    else:
        tokenSound.play()


    for i in range(len(winnerRank)):
        rank = FONT.render(f'Position :{i+1}.', True, (0, 0, 0))
        screen.blit(rank, (600, 85 + (40*i)))
        screen.blit(color[winnerRank[i]], (620, 75 + (40*i)))

    pygame.display.update()
    time.sleep(0.05)


def show_all():

    for i in SAFE[4:]:
        screen.blit(star, i)

    for i in range(len(position)):
        for j in position[i]:
            screen.blit(color[i], j)

    screen.blit(DICE[number-1], DicePosition[currentPlayer])

    for i in range(len(winnerRank)):
        rank = FONT.render(f'{i+1}.', True, (0, 0, 0))
        screen.blit(rank, (600, 85 + (40*i)))
        screen.blit(color[winnerRank[i]], (620, 75 + (40*i)))
    
def is_possible(x, y):
    if position[x][y] in WINNER:
        return False

    if (position[x][y][1] == 284 and position[x][y][0] <= 202 and x == 0) \
            and (position[x][y][0] + 38*number > WINNER[x][0]):
        return False

    elif (position[x][y][1] == 284 and 368 < position[x][y][0] and x == 2) \
            and (position[x][y][0] - 38*number < WINNER[x][0]):
        return False
    elif (position[x][y][0] == 284 and position[x][y][1] <= 202 and x == 1) \
            and (position[x][y][1] + 38*number > WINNER[x][1]):
        return False
    elif (position[x][y][0] == 284 and position[x][y][1] >= 368 and x == 3) \
            and (position[x][y][1] - 38*number < WINNER[x][1]):
        return False
    return True

def move_token(x, y):
    global currentPlayer, diceRolled

    if tuple(position[x][y]) in HOME[currentPlayer] and number == 6:
        position[x][y] = list(SAFE[currentPlayer])
        tokenSound.play()
        diceRolled = False

    elif tuple(position[x][y]) not in HOME[currentPlayer]:
        diceRolled = False
        if not number == 6:
            currentPlayer = (currentPlayer+1) % 4

        for i in range(number):
            if (position[x][y][1] == 284 and position[x][y][0] <= 202 and x == 0) \
                    and (position[x][y][0] + 38 <= WINNER[x][0]): 
                    position[x][y][0] += 38
                    show_token(x, y)

            elif (position[x][y][1] == 284 and 368 < position[x][y][0] and x == 2) \
                    and (position[x][y][0] - 38*number >= WINNER[x][0]):
                    position[x][y][0] -= 38
                    show_token(x,y)

            elif (position[x][y][0] == 284 and position[x][y][1] <= 202 and x == 1) \
                    and (position[x][y][1] + 38*number <= WINNER[x][1]):
                # for i in range(number):
                    position[x][y][1] += 38
                    show_token(x,y)
            #  B2
            elif (position[x][y][0] == 284 and position[x][y][1] >= 368 and x == 3) \
                    and (position[x][y][1] - 38*number >= WINNER[x][1]):
                # for i in range(number):
                    position[x][y][1] -= 38
                    show_token(x,y)

        # Other Paths
            else:
                #  R1, Y3
                if (position[x][y][1] == 240 and position[x][y][0] < 202) \
                        or (position[x][y][1] == 240 and 368 <= position[x][y][0] < 558):
                    position[x][y][0] += 38
                # R3 -> R2 -> R1
                elif (position[x][y][0] == 12 and position[x][y][1] > 240):
                    position[x][y][1] -= 44

                #  R3, Y1
                elif (position[x][y][1] == 328 and 12 < position[x][y][0] <= 202) \
                        or (position[x][y][1] == 328 and 368 < position[x][y][0]):
                    position[x][y][0] -= 38
                #  Y3 -> Y2 -> Y1
                elif (position[x][y][0] == 558 and position[x][y][1] < 328):
                    position[x][y][1] += 44

                #  G3, B1
                elif (position[x][y][0] == 240 and 12 < position[x][y][1] <= 202) \
                        or (position[x][y][0] == 240 and 368 < position[x][y][1]):
                    position[x][y][1] -= 38
                # G3 -> G2 -> G1
                elif (position[x][y][1] == 12 and 240 <= position[x][y][0] < 328):
                    position[x][y][0] += 44

                #  B3, G1
                elif (position[x][y][0] == 328 and position[x][y][1] < 202) \
                        or (position[x][y][0] == 328 and 368 <= position[x][y][1] < 558):
                    position[x][y][1] += 38
                #  B3 -> B2 -> B1
                elif (position[x][y][1] == 558 and position[x][y][0] > 240):
                    position[x][y][0] -= 44
                
                else:
                    for i in jump:
                        if position[x][y] == list(i):
                            position[x][y] = list(jump[i])
                            break

                show_token(x, y)

        if tuple(position[x][y]) not in SAFE:
            for i in range(len(position)):
                for j in range(len(position[i])):
                    if position[i][j] == position[x][y] and i != x:
                        position[i][j] = list(HOME[i][j])
                        killSound.play()
                        currentPlayer = x # (currentPlayer+3) % 4

def check_winner():
    global currentPlayer
    if currentPlayer not in winnerRank:
        for i in position[currentPlayer]:
            if i not in WINNER:
                return
        winnerRank.append(currentPlayer)
    else:
        currentPlayer = (currentPlayer + 1) % 4


old_winner_rank = []

while True:
    game_type = input("Enter the type of gameplay (1 for all AI players, 2 for 1 human player and 3 AI players, 3 for 2 human players and 2 AI players, 4 for 3 human players and 1 AI player): ")
    game_type = int(game_type)
    if game_type in [1, 2, 3, 4]:
        break


running = True
while(running):

    screen.fill((0, 0, 0))
    screen.blit(board, (0, 0)) 

    check_winner()

    for event in pygame.event.get():

        if event.type == pygame.QUIT or (event.type== pygame.KEYUP and event.key==K_ESCAPE):
            running = False

        if event.type == pygame.MOUSEBUTTONUP and game_type != 1:
            coordinate = pygame.mouse.get_pos()

            if game_type == 2 and currentPlayer != 0 or game_type == 3 and currentPlayer != 0 and currentPlayer != 1 or game_type == 4 and currentPlayer != 0 and currentPlayer != 1 and currentPlayer != 2:
                pass
            else:
                if not diceRolled and (DicePosition[currentPlayer][1] <= coordinate[1] <= DicePosition[currentPlayer][1]+49) and (DicePosition[currentPlayer][0] <= coordinate[0] <= DicePosition[currentPlayer][0]+49):
                    number = random.randint(1, 6)
                    diceSound.play()
                    flag = True
                    for i in range(len(position[currentPlayer])):
                        if tuple(position[currentPlayer][i]) not in HOME[currentPlayer] and is_possible(currentPlayer, i):
                            flag = False
                    if (flag and number == 6) or not flag:
                        diceRolled = True

                    else:
                        currentPlayer = (currentPlayer+1) % 4

                elif diceRolled:
                    for j in range(len(position[currentPlayer])):
                        if position[currentPlayer][j][0] <= coordinate[0] <= position[currentPlayer][j][0]+31 \
                                and position[currentPlayer][j][1] <= coordinate[1] <= position[currentPlayer][j][1]+31:
                            move_token(currentPlayer, j)
                            break
        
    if game_type == 1 or game_type == 2 and currentPlayer != 0 or game_type == 3 and currentPlayer != 0 and currentPlayer != 1 or game_type == 4 and currentPlayer != 0 and currentPlayer != 1 and currentPlayer != 2:
            
        if not diceRolled:
            number = random.randint(1, 6)
        
            diceSound.play()
            flag = True
            for i in range(len(position[currentPlayer])):
                if tuple(position[currentPlayer][i]) not in HOME[currentPlayer] and is_possible(currentPlayer, i) and tuple(position[currentPlayer][i]) not in WINNER:
                    flag = False
            if (flag and number == 6) or not flag:
                diceRolled = True

            else:
                currentPlayer = (currentPlayer+1) % 4

        elif diceRolled:
        
            old_state = {
                'positions': [list(player) for player in position]  
            }
            old_state_processed = preprocess_state(old_state)
            
            if currentPlayer == 0:
                q_network = r_q_network
                optimizer = r_optimizer
            elif currentPlayer == 1:
                q_network = g_q_network
                optimizer = g_optimizer
            elif currentPlayer == 2:
                q_network = y_q_network
                optimizer = y_optimizer
            elif currentPlayer == 3:
                q_network = b_q_network
                optimizer = b_optimizer

            action = choose_action(old_state_processed, q_network)

            while ((tuple(position[currentPlayer][action]) in HOME[currentPlayer] and number != 6 ) or not is_possible(currentPlayer, action))or position[currentPlayer][action] in WINNER or (position[currentPlayer][action] in winner_path[currentPlayer] and number == 6):
            
                reward = -10
                q_values = q_network(torch.tensor(old_state_processed, dtype=torch.float32))
                next_q_values = q_network(torch.tensor(old_state_processed, dtype=torch.float32))
                print(f"Q-Values shape: {q_values.shape}, Next Q-Values shape: {next_q_values.shape}, Action: {action}")
                
                q_values = q_values.unsqueeze(0)
                next_q_values = next_q_values.unsqueeze(0)
                
                max_next_q = torch.max(next_q_values).item()
                
                target_q = q_values.clone()
                target_q[0][action] = reward + 0.9 * max_next_q  # Discount factor
                
                loss = criterion(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                flag_two = False
                action = choose_action(old_state_processed, q_network)

                for i in range(len(position[currentPlayer])+1):
                    if i == len(position[currentPlayer]) and number == 6:
                        flag_two = True
                    if (position[currentPlayer][i]) in winner_path[currentPlayer]:
                        continue
                    else:
                        break
                if flag_two:
                    break

            move_token(currentPlayer, action)

            new_state = {
                    'positions': [list(player) for player in position]  # Updated positions
                    }
            
            new_state_processed = preprocess_state(new_state)
            
            print(f"Old State: {old_state}, New State: {new_state}, Action Taken: {action}")

            if 0 <= action < len(old_state) and 0 <= action < len(new_state):
                reward = calculate_reward(old_state, new_state, action, playerKilled, winnerRank, old_winner_rank)
            else:
                print("Invalid action index")
                reward = 0  


            
            reward = calculate_reward(old_state, new_state, action, playerKilled, winnerRank, old_winner_rank)

            q_values = q_network(torch.tensor(old_state_processed, dtype=torch.float32))
            next_q_values = q_network(torch.tensor(new_state_processed, dtype=torch.float32))
            print(f"Q-Values shape: {q_values.shape}, Next Q-Values shape: {next_q_values.shape}, Action: {action}")

            q_values = q_values.unsqueeze(0)
            next_q_values = next_q_values.unsqueeze(0)

            max_next_q = torch.max(next_q_values).item()

            target_q = q_values.clone()
            target_q[0][action] = reward + 0.9 * max_next_q  # Discount factor

            loss = criterion(q_values, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    old_winner_rank = winnerRank.copy()
        
    show_all()

    pygame.display.update()

    pygame.time.delay(300)



print("Saving model...")
torch.save(r_q_network.state_dict(), r_model_path)
torch.save(g_q_network.state_dict(), g_model_path)
torch.save(y_q_network.state_dict(), y_model_path)
torch.save(b_q_network.state_dict(), b_model_path)

pygame.quit()