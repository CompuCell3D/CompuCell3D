Documentation For Python batch testing script(ie. batch_test_init.py)
Created: 10/5/2018 Author: Benjamin Siefers

Feel free to contact me about questions about this script at ben@siefers.com.

This script tests all the .cc3d scripts in a specific  directory and collects output whether 
the scripts have failed or succeeded the output can by default is in the script’s home directory 
unless specified through a command line argument. You can collect output if you wish, but unless 
it is necessary I reccommend to use the --noOutput command line argument, as the files the demos 
create can be VERY(>1 TB) large depending on how many demos ran and the time it takes for each 
demo are run. The script was made using Python 2.7.15 use your appropriate 2.7.* Python command 
to run the script. Go to where the batch_test_init.py file is located use the Python command 
call the script in addition to the command arguments you wish to use.

Command Line Arguments:

The script extends most of CompuCell3D’s command arguments including -player. The script calls the 
compucell3d.sh file where the extension depends on the type of system you are using. The file can be 
found in CompuCell3D’s top directory. The default type of script ran is the runScript.sh file found 
with the compucell3d.sh file in the top directory.

-h / --help:

Shows a list of commands and a brief explanation of the script’s purpose.                                     

--noOutput:

Tells Compucell3D to not store screenshots or vtk files. Test results will still be generated.

-i [Input Directory]:

The command argument tells the script where to find the directory that has the demos that need to 
be ran. By default, the script uses the Demos directory.
Don’t worry about files that are not a .cc3d file they will be ignored.
The command argument takes a directory as opposed to a file location in CompuCell3D’s -i command.

-o [Output Directory Path]:

The command argument specifies which directory to send the screenshots or vtk file output.

-testOutput [Test Result Path]:

The command argument specifies which directory to send the SuccessfulResults-M-D-Y_H_M.txt, 
UnexpectedResults-M-D-Y_H_M.txt,and ProcessedDemos-M-D-Y_H_M.txt file. The path is expected to 
already exist.

--player:
Uses the player if you want to have screenshots opposed to vtk files.

Examples:

python batch_test_init.py -i [InputDirectory] -o [OutputDirectory] -testOutput [TestOutputDirectory] --player

python batch_test_init.py #defaults to using Demos directory and sends Demo outputs to CC3DWorkspace

python batch_test_init.py --noOutput #stores test results from the demo directory, but not the demo data
