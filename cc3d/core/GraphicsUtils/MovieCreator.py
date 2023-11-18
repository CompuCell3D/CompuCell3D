# -*- coding: utf-8 -*-
import os
import subprocess
import tempfile
import shutil

def makeMovie(simulationPath, frameRate, quality):
    """
    :param simulationPath: a string path to a directory with a .cc3d file and screenshot directories
    :param frameRate: an int >= 1
    :param quality: an int 1-10 (inclusive)
    :return: the number of movies created
    """
    #Credit to https://stackoverflow.com/q/49581846/16519580 user 'Makes' for the text overlay FFMPEG command.

    if not os.path.exists(simulationPath):
        print(f"Error: Could not make movie inside unknown directory `{simulationPath}`")
        return 0

    print("Making movie inside `", simulationPath, "`")
    movieCount = 0

    for visualizationName in os.listdir(simulationPath):
        inputPath = os.path.join(simulationPath, visualizationName)
        if not os.path.isdir(inputPath):
            continue

        # 'delete=True' removes the temporary file when it is closed.
        # So, setting 'delete=True' ensures that the tempfile stays active long enough for FFMPEG to read it.
        with tempfile.NamedTemporaryFile(delete=False, mode='+a', dir=inputPath) as temp_file:
            with tempfile.NamedTemporaryFile(delete=False, mode='+a', dir=inputPath) as text_overlay_file:

                frameCount = 0
                for fileName in os.listdir(inputPath):
                    if fileName.lower().endswith(".png"):
                        temp_file.write(f"file '{fileName}'\n")
                        text_overlay_file.write(f"{frameCount} drawtext reinit 'text=MCS {frameCount}';\n")
                        frameCount += 1
                temp_file.close()
                text_overlay_file.close()
                print("Wrote to text_overlay_file",text_overlay_file.name)

                if frameCount > 0:
                    # Number the file name so that it does not overwrite another movie
                    fileNumber = 0
                    outputPath = os.path.join(simulationPath, "movies")
                    subprocess.run([
                        "mkdir", outputPath
                    ])

                    while os.path.exists(os.path.join(outputPath, f"{visualizationName}_{fileNumber}.mp4")):
                        fileNumber += 1
                    outputPath = os.path.join(outputPath, f"{visualizationName}_{fileNumber}.mp4")

                    subprocess.run([
                        "ffmpeg",
                        "-n",  # never overwrite a file
                        "-r", str(frameRate),  # output frame rate
                        "-f", "concat",
                        "-safe", "0",
                        "-i", temp_file.name,
                        "-crf", str(quality),  # set quality (constant rate factor, crf): 51=worst, 0=best
                        "-c:v", "libx264",  # video codec: H.264
                        "-pix_fmt", "yuv420p",
                        "-filter_complex", f"[0:v]sendcmd=f={os.path.basename(text_overlay_file.name)},drawtext=fontfile=PF.ttf:text='':fontcolor=white:fontsize=20",
                        outputPath
                    ], cwd=inputPath)

                    if os.path.exists(outputPath):
                        movieCount += 1

                os.remove(text_overlay_file.name)
            os.remove(temp_file.name)

    print(f"Created {movieCount} movies inside `{simulationPath}`")
    return movieCount


def makeMovieWithSettings():
    import cc3d.player5.Configuration as Configuration

    # Choose the most recently modified subdir of the project dir
    projectPathRoot = Configuration.getSetting("OutputLocation")
    maxLastModifiedTime = 0
    simulationPath = None
    for dirName in os.listdir(projectPathRoot):
        dirPath = os.path.join(projectPathRoot, dirName)
        if os.path.isdir(dirPath):
            if os.path.getmtime(dirPath) > maxLastModifiedTime:
                maxLastModifiedTime = os.path.getmtime(dirPath)
                simulationPath = dirPath

    frameRate = Configuration.getSetting("FrameRate")
    quality = Configuration.getSetting("Quality")

    return makeMovie(simulationPath, frameRate, quality)
