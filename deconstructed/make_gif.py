from PIL import Image, ImageDraw, ImageFont
import os
import argparse
import imageio


def rename_images(directory,filename):
    f = os.path.join(directory, filename)
    filenameSplit = filename.split("_")
    alg = f"{filenameSplit[0]}_{filenameSplit[1]}_{filenameSplit[2]}"
    fileType = filenameSplit[-2]
    iterSplit = filenameSplit[-1].split(".")
    iter = iterSplit[0]
    return alg, fileType, iter, f

def resize_image(img):
    mode = img.mode
    oldWidth = img.width
    oldHeight = img.height
    newHeight = img.height + 100
    newWidth = img.width + 0
    x1 = 0
    y1 = 0
    newImage = Image.new(mode, (newWidth,newHeight), "WHITE")
    newImage.paste(img, (x1,y1,x1+oldWidth,y1+oldHeight))
    return newImage, newHeight, newWidth

def write_image_text(resizedImage, resizedHeight, resizedWidth, alg, iter, fnt):
    d = ImageDraw.Draw(resizedImage)
    d.text((60, resizedHeight - 60), f"{alg}", font=fnt, fill=(0, 0, 0))
    d.text((resizedWidth - 200, resizedHeight - 60), f"{iter}", font=fnt, fill=(0, 0, 0))

def get_folder_name(directory):
    dirSplit = directory.split("/")
    folderName = dirSplit[-1]
    return folderName

def make_gif_from_PNG(directory, log_freq, gif_files):
    images = []
    directory = os.path.join(directory, gif_files)
    files = os.listdir(directory)
    files.sort()

    folderName = get_folder_name(directory)
    
    iter = 0
    for filename in files:
        f = os.path.join(directory, filename)
        if (iter % log_freq) == 0:
            images.append(imageio.imread(f))
        iter = iter + 1

    print(f"{directory}/heatmap_{folderName}_{iter}_{log_freq}.gif")
    imageio.mimsave(f"{directory}/heatmap_{folderName}_{iter}_{log_freq}.gif", images)

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def main(outdir_path):
    directory = outdir_path
    fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 35)
    gif_files = "gif_files"
    if os.path.isdir(directory):
        if not os.path.isdir(f"{directory}/{gif_files}"):
            os.mkdir(f"{directory}/{gif_files}")

    files = os.listdir(directory)
    files.sort()
    
    for filename in files:
        if 'heatmap' in filename and 'edit' not in filename:
            alg, fileType, iter, f = rename_images(directory,filename)
            img = Image.open(f)#.convert("RGBA")
            resizedImage, resizedHeight, resizedWidth = resize_image(img)
            write_image_text(resizedImage, resizedHeight, resizedWidth, alg, iter, fnt)
            resizedImage.save(f"{directory}/{gif_files}/{fileType}_{iter}_{alg}.png")


    log_freq = 1
    make_gif_from_PNG(directory, log_freq, gif_files)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Collect images, label them, and save them in gif_files subdirectory.')
    parser.add_argument('--path', type=dir_path)
    args = parser.parse_args()
    
    directory = args.path
    fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 35)
    gif_files = "gif_files"
    if os.path.isdir(directory):
        if not os.path.isdir(f"{directory}/{gif_files}"):
            os.mkdir(f"{directory}/{gif_files}")

    files = os.listdir(directory)
    files.sort()
    
    for filename in files:
        if 'heatmap' in filename and 'edit' not in filename:
            alg, fileType, iter, f = rename_images(filename)
            img = Image.open(f)#.convert("RGBA")
            resizedImage, resizedHeight, resizedWidth = resize_image()
            write_image_text(resizedImage, resizedHeight, resizedWidth)
            resizedImage.save(f"{directory}/{gif_files}/{fileType}_{iter}_{alg}.png")


    log_freq = 1
    make_gif_from_PNG(directory, log_freq)









"""
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        print(f)
        img = Image.open(f)
        d = ImageDraw.Draw(img)
        d.text((20, img.height -20), f"{filename}", font=fnt, fill=(255, 255, 0))
        img.save(f"{directory}/{filename}_edit.png")


    #print(directory)
    #f = os.path.join(directory, filename)
    # checking if it is a file
    #if os.path.isfile(f):
    #    print(f)
 
#img = Image.new('RGB', (100, 30), color = (73, 109, 137))


#img = Image.open()
#fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 15)
#d = ImageDraw.Draw(img)
#d.text((10,10), "Hello world", font=fnt, fill=(255, 255, 0))
 
#img.save('pil_text_font.png')
"""