import os
import fcntl
import time
from modules.file_utils import FileUtils


class CommandTxtUtils(object):

    FILENAME = './commands.txt'

    def __init__(self):
        self.commands = []

        if os.path.exists(self.FILENAME):
            with open(self.FILENAME, 'r') as outfile:
                FileUtils.lock_file(outfile)
                self.commands = outfile.readlines()
                FileUtils.unlock_file(outfile)


    def is_stopped(self, args):
        for command in self.commands:
            command = command.strip()
            if command == 'all' or command == 'id=' + str(args.id) \
                    or command == 'repeat_id=' + str(args.repeat_id) \
                    or command == 'report=' + str(args.report):
                return True
        return False