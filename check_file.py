import SimpleITK as sikt
import os
dir = "/home/dd/flare2022/data/FLARE22_UnlabeledCase251-500/"
for x in os.listdir(dir):
    try:
        ( sikt.GetArrayFromImage(sikt.ReadImage(dir + x)).shape)
    except Exception:
        os.remove(dir + x)
        print(x)