# board_dict = {0:[], 1}

#should board be a class? maybe with top and board
from __future__ import print_function
import random
import os
import neat
import visualize

HEIGHT = 5
WIDTH = 3
EMPTY_ROW = [0 for i in range(WIDTH)]
FULL_ROW = [1 for i in range(WIDTH)]

action_history = []

# I, O, T, S, Z, J, and L
pieces = [  [1,1,1,1], #L
            [[1,1],[1,1]],#O
            [[1,1,1],[0,1,0]],#T
            [[0,1,1],[1,1,0]],#S
            [[1,1,0],[0,1,1]],#Z
            [[1,1,1],[0,0,1]],#J
            [[0,0,1],[1,1,1]]]#L





def new_board():
    return [[0 for i in range(WIDTH)] for j in range(HEIGHT)]

def top_height():
    return [0 for i in range(WIDTH)]

def text_render(board):
    for row in board:
        print(row)

def rotate_piece(piece, orientation):
    pass
    #actions are mirror(2) and x/y shift(1,3)
    # 001 10 111 11 
    # 111 10 100 01
    #     11     01
    
    # 0   p(0,0)=p'(

def create_piece():
    # 7bag
    # random.shuffle([n for range(7)])
    return random.choice(pieces)
    # return random.randint(0,(len(pieces)-1))

def update_top():
    pass

def clear_row(board):
    # Identify how many rows are removed and copy rows down.
    drop = 0
    for i in reversed(range(len(board))):
        if board[i] == FULL_ROW: #all(e == 1 for e in row): #checks that the whole row is full
            drop += 1
            board[i] = [0 for i in range(WIDTH)]
        elif drop:
            board[i+drop] = board[i]
            board[i] = [0 for i in range(WIDTH)]
    
    return drop

def drop_piece(board, highest_row, piece, orientation):
    # create check chart and run

    return board, highest_row

def new_row():
    return [0.0 for i in range(WIDTH)]


def tetris_game():
    run = True
    board = new_board()
    highest_row = [0 for i in range(WIDTH)]
    piece = [[1]]
    score = 0
    while run:
        os.system('clear')
        text_render(board)
        # print(highest_row)0
        col = int(input(f"which row 0-{WIDTH-1}?"))
        # piece_width = len(piece[0])
        # for i in range(piece_width)):0
        #     piece[piece_width-1][i]
        board[HEIGHT-highest_row[col]-1][col] = 1
        highest_row[col] += 1
        if highest_row[col] > HEIGHT:
            print("LOSE")
            break
        drop = clear_row(board)
        score += drop
        for i in range(len(highest_row)):
            highest_row[i] -= drop


def tetris_eval_1b(genomes, config):
    #initialize new game
    # run = True
    # board = new_board()
    highest_row = new_row()
    score = 0

    #  loop - generate piece, print user info, ask for input, rotate, drop, check/update for clears, update toplist. 

    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        while score < 100:
            tet_output = net.activate(tuple(highest_row))
            i = tet_output.index(max(tet_output))
            highest_row[i] += 1.0
            if highest_row[i] > HEIGHT:
                break
            if all(highest_row):
                for i in range(WIDTH):
                    highest_row[i] -= 1.0
                genome.fitness += 1
                score += 1

        # for xi, xo in zip(xor_inputs, xor_outputs):
        #     output = net.activate(xi)
        #     genome.fitness -= (output[0] - xo[0]) ** 2

# tetris_game()

# """
# 2-input XOR example -- this is most likely the simplest possible example.
# """

# from __future__ import print_function
# import os
# import neat
# import visualize

# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [   (0.0,),     (1.0,),     (1.0,),     (0.0,)]

example_inputs = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0), (2.0, 0.0, 1.0),
                (0.0, 1.0, 0.0), (0.0, 2.0, 0.0), (1.0, 2.0, 0.0), (0.0, 2.0, 1.0), 
                (0.0, 0.0, 1.0), (0.0, 0.0, 2.0), (1.0, 0.0, 2.0), (0.0, 1.0, 2.0), 
                (2.0, 2.0, 0.0), (2.0, 0.0, 2.0), (0.0, 2.0, 2.0)]


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = net.activate(xi)
            genome.fitness -= (output[0] - xo[0]) ** 2

# Boilerplate from xor
def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 100 generations.
    # winner = p.run(eval_genomes, 100)
    winner = p.run(tetris_eval_1b, 100)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for c in example_inputs:
            output = winner_net.activate(c)
            print(f"example state {c} choosey moms choose {output}")
            # output = winner_net.activate(xi)
            # print("example state {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    node_names = {-1:'in 0', -2: 'in 1', -3: 'in 2', 0: 'out 0', 1: 'out 1', 2: 'out 2'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 10)

# Boilerplate from xor
if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)