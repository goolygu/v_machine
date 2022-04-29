Visual Loop Machine
###################

Visual Loop Machine plays visual loops stored in the MTD (Multiple Temporal Dimension) video format. The visual loop
changes according to the loudness of the audio playing on the same computer.

Example videos:
 | https://youtu.be/9IMoNuqwvhs
 | https://youtu.be/jDYyhgoLwZ0

Quickstart
==========
Create virtual environment and install v_machine

.. code-block:: console

    $ git clone https://github.com/goolygu/v_machine.git
    $ cd v_machine
    $ make venv

Place mtd videos to play under the mtd_video folder (Samples are provided.) You can download mtd videos
created by me here https://drive.google.com/drive/folders/16wlG6fFPS-srPqVNeYKTvZyl0b4hTfPi?usp=sharing

Activate virtual environment and run the following command to start

.. code-block:: console

    $ source venv/bin/activate
    $ python src/v_machine/v_machine.py
