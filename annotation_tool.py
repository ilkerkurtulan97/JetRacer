import os
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

class ImageAnnotator:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Annotator")

        self.label = Label(master, text="Select an image folder:")
        self.label.pack()

        self.load_button = Button(master, text="Load Folder", command=self.load_folder)
        self.load_button.pack()

        self.canvas = Canvas(master, width=224, height=224, bg='white')
        self.canvas.pack()

        self.image_list = []
        self.current_image = None
        self.current_img_path = ""
        self.index = 0

    def load_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.image_list = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
            self.index = 0
            self.load_image()
        else:
            print("No folder selected")

    def load_image(self):
        if self.index < len(self.image_list):
            self.current_img_path = self.image_list[self.index]
            img = Image.open(self.current_img_path)
            img = img.resize((224, 224), Image.LANCZOS)
            self.current_image = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, image=self.current_image, anchor=NW)
            self.canvas.bind("<Button-1>", self.click_event)
        else:
            self.canvas.unbind("<Button-1>")
            print("All images have been processed.")

    def click_event(self, event):
        x, y = event.x, event.y
        directory, filename = os.path.split(self.current_img_path)
        name_parts = filename.split('_')
        new_filename = f"{x}_{y}_" + "_".join(name_parts[2:])
        new_path = os.path.join(directory, new_filename)
        os.rename(self.current_img_path, new_path)
        print(f"Renamed to {new_path}")
        self.index += 1
        self.load_image()

if __name__ == "__main__":
    root = Tk()
    annotator = ImageAnnotator(root)
    root.mainloop()
