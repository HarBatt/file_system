import PIL
from PIL import Image

class Processing(object):
    def __init__(self, path):
        self.path = path
        self.img = Image.open(self.path)
    
    def resize(self, width):
        wpercent = (width/float(self.img.size[0]))
        hsize = int((float(self.img.size[1])*float(wpercent)))
        new_img = self.img.resize((width, hsize), PIL.Image.ANTIALIAS)
        return new_img
    
    def resize_save(self, updated_path):
        img = self.resize(128)
        img.save(updated_path)
