# -*- coding: utf-8 -*-
"""
Contains an inhibitor specific to the Windows OS to ensure
that the operating systme does not sleep with it is running.py

A few functions and classes for cmd line output
"""
import os
import sys
import threading
import glob

class WindowsInhibitor:

    '''Prevent OS sleep/hibernate in windows'''

    ES_CONTINUOUS = 0x80000000

    ES_SYSTEM_REQUIRED = 0x00000001


    def __init__(self):

        pass

    def inhibit(self):
        """start the windows inhibitor"""

        import ctypes

        print("\nPreventing Windows from going to sleep \n")

        ctypes.windll.kernel32.SetThreadExecutionState(

            WindowsInhibitor.ES_CONTINUOUS | \

            WindowsInhibitor.ES_SYSTEM_REQUIRED)

    def uninhibit(self):
        """stop the windows inhibitor"""

        import ctypes

        print("\nAllowing Windows to go to sleep")

        ctypes.windll.kernel32.SetThreadExecutionState(

            WindowsInhibitor.ES_CONTINUOUS)

# =============================================================================
def banner(text, chrc='=', length=70):
    """prints a banner to the cmd line"""
    spaced_text = ' {} '.format(text)
    ban = spaced_text.center(length, chrc)

    print("")
    print(ban)
# =============================================================================
def calculate_file_progress(path, final, extension=None):
    """calculates and prints the % of files in a directory"""

    files_in_path = glob.glob(os.path.join(path, "*.{}".format(extension)))
    no_files = len(files_in_path)
    perc_done = int(round(no_files/final * 100))

    print(perc_done, "%")
# =============================================================================
class ProgressBar(threading.Thread):
    """
     In a separate thread, print dots to the screen until terminated.
    """
    def __init__(self):
        threading.Thread.__init__(self)
        self.event = threading.Event()

    def run(self):
        """run and print the dots to the cmd line"""
        event = self.event # make local
        while not event.is_set():
            sys.stdout.write(".")
            sys.stdout.flush()
            event.wait(5) # pause for 5 seconds
            sys.stdout.write(".")

    def stop(self):
        """stop the progress dots"""
        self.event.set()
