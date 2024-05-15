from PIL import Image

class BadNetTriggerHandler(object):
    def __init__(self, trigger_label, trigger_path, trigger_size, img_width, img_height):
        self.trigger_label = trigger_label
        self.trigger_img = Image.open(trigger_path).convert('RGB')
        self.trigger_size = trigger_size
        self.trigger_img = self.trigger_img.resize((trigger_size, trigger_size))        
        self.img_width = img_width
        self.img_height = img_height

    def put_trigger(self, img):
        img.paste(self.trigger_img, (self.img_width - self.trigger_size, self.img_height - self.trigger_size))
        return img
