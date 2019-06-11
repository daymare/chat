

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
    any key - continue conversation
    s - toggle show full conversation or exchange by exchange
    q - quit

    TODO implement help message

    """

    i = 0
    quit = False
    show = True

    while quit is False:
        os.system('clear')
        print("chat {}".format(i))
        print("show mode: {}".format(show))

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
            print("", flush=True, end="")

            # take next command
            if show is False:
                ch = getch()

                if ch == "n":
                    # auto toggle will handle it for us
                    i += 0
                    break
                elif ch == "p":
                    # need to go back twice as 1 will be added
                    i -= 2
                    break
                elif ch == "q":
                    quit = True
                    break
                elif ch == "s":
                    show = True

            print("        ", end="")
            print_sentence(your_statement)
            print("", flush=True, end="")

        if show is True:
            ch = getch()

            if ch == "n":
                i += 1
            elif ch == "p":
                i -= 1
            elif ch == "q":
                quit = True
            elif ch == "s":
                show = False
        else:
            i += 1

        # ensure i is in a valid range
        i = i % len(dataset)





