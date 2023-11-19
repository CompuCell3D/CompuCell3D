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
    # Credit to https://stackoverflow.com/q/49581846/16519580 user 'Makes' for the text overlay FFMPEG command.
    # Credit to https://superuser.com/a/939386 uer 'llogan' for the text positioning in the FFMPEG command.

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
        with tempfile.NamedTemporaryFile(delete=False, mode='+a', dir=inputPath) as tempFile:
            with tempfile.NamedTemporaryFile(delete=False, mode='+a', dir=inputPath) as textOverlayFile:
                """
                Write the names of files to be used as video frames to tempFile 
                and write the text to draw to textOverlayFile.
                """
                frameCount = 0
                duration = 1 / max(frameRate, 1)
                for fileNameExt in os.listdir(inputPath):
                    fileName, fileExtension = os.path.splitext(fileNameExt)
                    if fileExtension.lower() == ".png":
                        tempFile.write(f"file '{fileNameExt}'\n")

                        # Note: frameRate is excluded in the FFMPEG command.
                        # Instead, we use `duration` inside the input file.
                        # This fixes a bug where the last frame appears at the beginning.
                        tempFile.write(f"duration {duration}\n")

                        # Try to use the MCS listed in the screenshot name
                        mcs = 0
                        try:
                            parts = fileName.split("_")
                            if len(parts) >= 2:
                                mcs = int(parts[-1])
                        except:
                            mcs = frameCount

                        # Center the text
                        textOverlayFile.write(f"{frameCount} drawtext reinit 'text=MCS {mcs}':x=(w-text_w)/2:y=0;\n")
                        frameCount += 1
                tempFile.close()
                textOverlayFile.close()

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
                        "-f", "concat",
                        "-safe", "0",
                        "-i", tempFile.name,
                        "-crf", str(quality),  # set quality (constant rate factor, crf): 51=worst, 0=best
                        "-c:v", "libx264",  # video codec: H.264
                        "-pix_fmt", "yuv420p",
                        "-filter_complex", f"[0:v]sendcmd=f={os.path.basename(textOverlayFile.name)},drawtext=fontfile=PF.ttf:text='':fontcolor=white:fontsize=20",
                        outputPath
                    ], cwd=inputPath)

                    if os.path.exists(outputPath):
                        movieCount += 1

                os.remove(textOverlayFile.name)
            os.remove(tempFile.name)

    print(f"Created {movieCount} movies inside `{simulationPath}` with frame rate {frameRate} and quality {quality}/51.")
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
