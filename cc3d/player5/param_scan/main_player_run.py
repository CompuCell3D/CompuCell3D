import sys
from cc3d.player5.compucell3d import main as main_player

print('THIS IS MAIN PLAYER RUN')

print('sys.argv=', sys.argv)

main_player(sys.argv)

if __name__ == '__main__':
    print('main_player_run')
    print('THIS IS MAIN PLAYER RUN INSIDE')
