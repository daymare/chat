

import tty
import sys
import termios
import os


def getch():
    fd = sys.stdin.fileno()
    settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, settings)
    return ch

def print_sentence(sentence):
    for word in sentence:
        print(word, end="")
        print(" ", end="")
    print()


def look_at_data(dataset):
    """ interactively look at the dataset

    n - next conversation
    p - previous conversation
    r - random conversation
    q - quit

    TODO implement help message

    """

    i = 0
    quit = False

    while quit is False:
        os.system('clear')
        print("chat {}".format(i))

        # print out the current conversation
        chat = dataset[i]
        print("your persona: ")
        for sentence in chat.your_persona:
            print("    ", end="")
            print_sentence(sentence)

        print()
        print("conversation: ")
        for exchange in chat.chat:
            partner_statement = exchange[0]
            your_statement = exchange[1]

            print("    ", end="")
            print_sentence(partner_statement)
            print("        ", end="")
            print_sentence(your_statement)
            print("", flush=True)

        # take next command
        ch = getch()

        if ch == "n":
            i += 1
            i = min(i, len(dataset)-1)
        elif ch == "p":
            i -= 1
            i = max(i, 0)
        elif ch == "q":
            quit = True



