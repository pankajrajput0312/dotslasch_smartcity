
import argparse

# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
# parser.add_argument("--train", help="train-test")
# parser.add_argument("--dataset_path", help='provide path of dataset')
import argparse

# Adding optional argument
# parser.add_argument("--imgpath", help="Show Output")
# parser.add_argument("--train", help = "train-test")
# Read arguments from command line
args = parser.parse_args()

import tensorflow as tf
import os
import pathlib
import tkinter
import time
import datetime
import numpy as np
import cv2
from tkinter.messagebox import showerror
from tkinter.filedialog import askopenfilename, askdirectory
from matplotlib import image, pyplot as plt
from IPython import display
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.utils import plot_model
import warnings

"""# Build Generator

"""


warnings.filterwarnings("ignore")

# warnings.warn("second example of warning!")

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


# img = np.zeros((320,320,3))

# up_model = upsample(100,3 )
# result = up_model(np.expand_dims(img, axis = 0))
# print(result.shape)


def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample(128, 4),  # (batch_size, 64, 64, 128)
        downsample(256, 4),  # (batch_size, 32, 32, 256)
        downsample(512, 4),  # (batch_size, 16, 16, 512)
        downsample(512, 4),  # (batch_size, 8, 8, 512)
        downsample(512, 4),  # (batch_size, 4, 4, 512)
        downsample(512, 4),  # (batch_size, 2, 2, 512)
        downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample(512, 4),  # (batch_size, 16, 16, 1024)
        upsample(256, 4),  # (batch_size, 32, 32, 512)
        upsample(128, 4),  # (batch_size, 64, 64, 256)
        upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (batch_size, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


OUTPUT_CHANNELS = 3
generator = Generator()

# tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

LAMBDA = 100
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(disc_generated_output, gen_output, target):
    disc_generated_output = tf.convert_to_tensor(disc_generated_output, dtype=tf.float32)
    gen_output = tf.convert_to_tensor(gen_output, dtype=tf.float32)
    # target = tf.convert_to_tensor(target, dtype = tf.float32)
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, 0


def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


discriminator = Discriminator()
# tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = 'apartment_checkpoints'
try:
    os.mkdir(checkpoint_dir)
    os.mkdir(os.path.join(checkpoint_dir, "results"))
except Exception as e:
    print(e)
try:
    os.mkdir(os.path.join(checkpoint_dir, "results"))
except Exception as e:
    print(e)
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                  discriminator_optimizer=discriminator_optimizer,
#                                  generator=generator,
#                                  discriminator=discriminator)
# print("testing_checkpoints............................................................................................")
# # print(checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)))
# try:
#     checkpoint.restore("weights/program")
# except Exception as e:
#     print(e)





'''
Write model code here: create3 different model and see how it works

Model1: Footprint_design
Model2: program_design
Model3: furniture_design
'''

'''Footprint model'''
footprint_generator = Generator()
footprint_discriminaor = Discriminator()
footprint_checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=footprint_generator,
                                 discriminator=footprint_discriminaor)

# footprint_checkpoint.restore(os.path.join("weights","footprint"))
# try:
#     footprint_checkpoint.restore("weights/footprint")
#     print("footprint_weights_loaded")
# except Exception as e:
#     print(e)

'''program model'''
program_generator = Generator()
program_discriminator = Discriminator()
program_checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=program_generator,
                                 discriminator=program_discriminator)
try:
    program_checkpoint.restore("weights/program")
    print("program_weights_loaded")
except Exception as e:
    print(e)


'''Furniture model'''
furniture_generator = Generator()
furniture_discriminator = Discriminator()
furniture_checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=furniture_generator,
                                 discriminator=furniture_discriminator)
# try:
#     furniture_checkpoint.restore("weights/ckpt481-100")
#     print("furniture_weights_loaded")
# except Exception as e:
#     print(e)


'''create multiple model flow'''
models_flow = {
    "boundary_to_footprint":[True, False, False],
    "boundary_to_program":[True, True, False],
    "boundary_to_furniture":[True,True, True],
    "footprint_to_program":[False, True, False],
    "footprint_to_furniture":[False, True, True],
    "program_to_furniture": [False, False, True]
}
def replace_output():
    # img_path = "program.png"
    img2_update = cv2.imread("output.png")
    img2_update = cv2.cvtColor(img2_update, cv2.COLOR_BGR2RGB)
    img2_update = cv2.resize(img2_update, (300,300))
    cv2.waitKey(100)
    image2_update = Image.fromarray(img2_update)
    
    test2 = ImageTk.PhotoImage(image2_update)
    # label2 = tkinter.Label(image=test2)
    # label2.image = test2
    # label2.place(relx=0.6, rely=0.43)

    label2.configure(image=test2)
    label2.image=test2

def update_output(this_image):
    print("new_output_update")
    print("this_output image size before reshape", np.array(this_image).shape)
    output_image = cv2.resize(np.array(this_image), (300,300)) 
    print("shape of image after reshape", np.array(output_image).shape)
    cv2.imwrite("output.png", np.array(output_image)*255)
    replace_output()
    # output_image = Image.fromarray(np.array(output_image))
    
    # # output_image = ImageTk.PhotoImage(output_image)
    # test2 = ImageTk.PhotoImage(output_image)
    # label2 = tkinter.Label(image=test2)
    # label2.image = test2
    # label2.place(relx=0.6, rely=0.43)
    # label2.configure(image=output_image)
    # label2.image= output_image
def show_output(test_input, prediction, visualize):
    # print(electronic_weight['weight'])
    # checkpoint.restore(electronic_weight['weight'])
    # prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))
    update_output(prediction[0])
    # final_output = ImageTk.PhotoImage(prediction[0])
    # label2.image = final_output
    display_list = [test_input[0], prediction[0]]
    print("enter into a big show output")

    # print(display_list[0])
    display_list[1] = display_list[1] * 0.5 + 0.5
    # print(display_list[1])

    title = ['Input Image', 'Predicted Image']
    # print(prediction)
    if visualize:
        for i in range(2):
            plt.subplot(1, 2, i + 1)
            plt.title(title[i])
            # Getting the pixel values in the [0, 1] range to plot.
            plt.imshow(display_list[i])
            plt.axis('off')
        # cv2.imwrite("output.png", prediction[0])
        plt.show()
        print("visualization")
    


def process_image():
    input_image_path = askopenfilename(parent=root, title="Select an image",
                                       initialdir='.', filetypes=[("Any", "*"), ("JPEG File", "jpg"), ("PNG File", "png")])
    if input_image_path.split('.')[-1] not in ['jpg', 'jpeg', 'png']:
        showerror('Error', 'Please select an image with .jpg or .png extension.')
        return
    img = cv2.imread(input_image_path)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    show_output(generator, img)


def preprocess_image(img):
    # img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def build_network(model_option, test_img, visualize):
    test_img = preprocess_image(test_img)
    input_img = test_img
    # print(input_img)
    models_possible = models_flow[model_option]
    if models_possible[0]:
        test_img = footprint_generator(test_img, training = True)
        print("image_passes_to_footprint")
    if models_possible[1]:
        if models_possible[0]:
            test_img = 1-test_img
        test_img = program_generator(test_img, training = True)
        print("image_passes_to_program")
    if models_possible[2]:
        test_img = furniture_generator(test_img, training = True)
        print("image_passes_to_furniture")
    output = test_img
    show_output(input_img, test_img, visualize)
    print("returning_output_here")
    return output


# build_network("program_to_furniture", "program1.png")

def predict_image():
    input_image = np.asarray(image1)
    test_img = np.zeros((300,300,3))
    test_img[...,0]=input_image
    test_img[...,1]=input_image
    test_img[...,2]=input_image
    print(clicked.get())
    print("enter into predict image")
    output_image = build_network(clicked.get(), test_img, False)
    print("back to predict image")
    # output_image = cv2.resize(output_image[0], (300,300)) 
    #     # = output_image
    # output_image = Image.fromarray(output_image)
    # output_image = ImageTk.PhotoImage(output_image)
    # label2.configure(image=output_image)
    # label2.image= output_image
    print("okay okay")

def generate():
    if test_type==1:
        test_on_image()
    else:
        predict_image()

def update_program_weights():
    try:
        program_checkpoint.restore(entry["program_weight_path"])
        print("program weights loaded now")
    except Exception as e:
        print(e)       


'''UPDATED GUI'''
import tkinter
from tkinter import *
from PIL import Image, ImageTk
import cv2
import numpy as np
from tkinter.messagebox import showerror
from tkinter.filedialog import askopenfilename, askdirectory



root = Tk()
root.geometry("1000x1000")

'''Create label of Generator software'''
Label(root, text = "Apartment GENERATOR", font = 'times 30 bold').place(relx= 0.3, rely = 0.05)


'''Create dropdown menu'''
options = [
    "footprint_to_program",
]
  
# datatype of menu text
clicked = StringVar()
  
# initial menu text
clicked.set( "footprint_to_program" )
lab = Label(root, width=22, text="Select_model_type: ", font = 'times 15 bold', anchor='w')
lab.place(relx=0.25, rely=0.15)
drop = OptionMenu( root , clicked , *options )
drop.place(relx = 0.6, rely = 0.15)


# select weights path
entry = {}
def update_footprint_path():
    weights_path = askopenfilename(parent=root, title="select weights",
                                       initialdir='.')
    entry['footprint_weight_path']= weights_path
    footprint_weights.set(weights_path)
    return

def update_program_path():
    weights_path = askopenfilename(parent=root, title="select weights",
                                       initialdir='.')
    entry['program_weight_path']= weights_path
    program_weights.set(weights_path)
    update_program_weights()
    return

def update_furniture_path():
    weights_path = askopenfilename(parent=root, title="select weights",
                                       initialdir='.')
    entry['furniture_weight_path']= weights_path
    furniture_weights.set(weights_path)
    
    print(entry)
# mEntry = Entry(mGui, textvariable = ment, text="bebe")




# ''' Select Footprint Model weights'''
# lab_footprint = Label(root, width=22, text="Footprint path:", font = 'times 15', anchor='w')
# lab_footprint.place(relx=0.15, rely=0.25)
# footprint_weights = StringVar()
# ent = Entry(root, textvariable=footprint_weights, width = 50, bd = 4)
# ent.place(relx=0.3, rely=0.25)
# entry["footprint_weight_path"] = ent
# # ents = entry
# # print(ents)
# # root.bind('<Return>', (lambda event, e = ents: fetch(e)))
# b1 = Button(root, text = 'select Footprint path',
#     command=update_footprint_path, bg= '#3F7FBF', bd=2)
# b1.place(relx=0.7, rely=0.25)


# '''Select Program Model Weights'''
# lab_program = Label(root, width=22, text="Program path: ", font = 'times 15', anchor='w')
# lab_program.place(relx=0.15, rely=0.30)
# program_weights = StringVar()
# ent_program = Entry(root, textvariable=program_weights, width = 50, bd = 4)
# ent_program.place(relx=0.3, rely=0.30)
# entry["program_weight_path"] = ent_program
# ents = entry
# print(ents)
# root.bind('<Return>', (lambda event, e = ents: fetch(e)))
# # root.config(bg = "red")

# b1 = Button(root, text = 'select Program path',
#     command=update_program_path, bg= '#3F7FBF', bd=2)
# b1.place(relx=0.7, rely=0.30)


# ''' Select Furniture Model weights'''
# lab_furniture = Label(root, width=22, text="Furniture path:", font = 'times 15', anchor='w')
# lab_furniture.place(relx=0.15, rely=0.35)
# furniture_weights = StringVar()
# ent_furniture = Entry(root, textvariable=furniture_weights, width = 50, bd = 4)
# ent_furniture.place(relx=0.3, rely=0.35)
# entry["furniture_weight_path"] = ent_furniture
# ents = entry
# print(ents)
# root.bind('<Return>', (lambda event, e = ents: fetch(e)))
# b1 = Button(root, text = 'select furniture path',
#     command=update_furniture_path, bg= '#3F7FBF', bd=2)
# b1.place(relx=0.7, rely=0.35)



# Create a photoimage object of the image in the path
# input and output Labels
input_label = Label(root, width=22, text="input_image", font = 'times 15 bold', anchor='w')
input_label.place(relx=0.2, rely=0.25)
output_label = Label(root, width=22, text="output_image", font = 'times 15 bold', anchor='w')
output_label.place(relx=0.7, rely=0.25)


img_path = "program.png"
## read both image in image1 and image2
img1 = np.zeros((300,300))
img2 = cv2.imread(img_path)
img2 = cv2.resize(img2, (300,300))


## Transform image in pillow format

# image1 = Image.open(img_path)
image1 = Image.fromarray(img1)
image2 = Image.fromarray(img2)
print(image1)


## create Tkinter image and then place them
# test1 = ImageTk.PhotoImage(image1)
# label1 = tkinter.Label(image=test1)
# label1.image = test1
# #placing label1
# label1.place(relx=0.1, rely=0.43)


test2 = ImageTk.PhotoImage(image2)
label2 = tkinter.Label(image=test2)
label2.image = test2
label2.place(relx=0.6, rely=0.3)

'''Create Canvas to draw footprint'''

canvas = Canvas(root, 
           width=300, 
           height=300, cursor='cross', bd = 0)
canvas.place(relx = 0.1, rely = 0.3)

img1 = np.zeros((300,300))
image1 = Image.fromarray(img1)
test1 = ImageTk.PhotoImage(image1)

from PIL import ImageDraw, Image
draw_image = ImageDraw.Draw(image1)



ix = -1
iy = -1
drawing = False
   

def on_button_press(event):
    global ix, iy, drawing, img 
    # save mouse drag start position
    start_x = event.x
    start_y = event.y
    drawing = True
    ix = start_x
    iy = start_y
    print("ix:  ", ix)
    print("iy:  ", iy)
    

color_white_black = {1: "#FFFFFF", 2: "#000000"}
def on_move_press(event):
    global ix, iy, drawing, img 
    if drawing:
        canvas.create_rectangle(ix, iy, event.x, event.y, fill =color_white_black[rectangle_color], width=0)
        draw_image.rectangle([ix,iy,event.x, event.y], fill=color_white_black[rectangle_color], width=0)

def on_button_release(event):
    global ix, iy, drawing, img 
    drawing = False
    canvas.create_rectangle(ix, iy, event.x, event.y, fill =color_white_black[rectangle_color], width=0 )
    draw_image.rectangle([ix,iy,event.x, event.y], fill=color_white_black[rectangle_color], width=0)    

canvas_image = canvas.create_image(0,0,anchor="nw",image=test1)
canvas.bind("<ButtonPress-1>", on_button_press)
canvas.bind("<B1-Motion>", on_move_press)
canvas.bind("<ButtonRelease-1>", on_button_release)


'''
    add select random image function and color selection function
'''

from PIL import Image

lab_color = Label(root, width=22, text="select_color: ", font = 'times 15 bold', anchor='w')
lab_color.place(relx=0.08, rely=0.7)
def selected_option():
    global rectangle_color
    rectangle_color = var.get()
    print(var.get())
    print(clicked.get())
var = IntVar()
rectangle_color = 1
R1 = Radiobutton(root, text="White", variable=var, value=1, bd = 2,  font = 'times 15', command = selected_option)
R1.place(relx = 0.2, rely = 0.7)

R2 = Radiobutton(root, text="Black", variable=var, value=2, bd = 2,  font = 'times 15', command = selected_option)
R2.place(relx = 0.3, rely = 0.7)

# print(var.get())

def update_image():
    global final_image 
    input_image_path = askopenfilename(parent=root, title="Select an image",
                                       initialdir='.', filetypes=[("Any", "*"), ("JPEG File", "jpg"), ("PNG File", "png")])
    if input_image_path.split('.')[-1] not in ['jpg', 'jpeg', 'png']:
        showerror('Error', 'Please select an image with .jpg or .png extension.')
        return
    img1 = np.zeros((300,300))
    update_input = cv2.imread(input_image_path)
    update_input = cv2.cvtColor(update_input, cv2.COLOR_BGR2RGB)
    update_input = cv2.resize(update_input, (300,300))
    update_input = Image.fromarray(update_input)
    test1 = ImageTk.PhotoImage(update_input)
    final_image = test1
    canvas.itemconfig(canvas_image,image=test1)
    # canvas.create_image(0,0,anchor="nw",image=test1)
    print("image_updated")

'''test on single image'''
def test_on_image():
    
    input_image_path = askopenfilename(parent=root, title="Select an image",
                                       initialdir='.', filetypes=[("Any", "*"), ("JPEG File", "jpg"), ("PNG File", "png")])
    if input_image_path.split('.')[-1] not in ['jpg', 'jpeg', 'png']:
        showerror('Error', 'Please select an image with .jpg or .png extension.')
        return
    img1 = np.zeros((300,300))
    input_image = cv2.imread(input_image_path)
    input_image = cv2.resize(input_image, (300,300))
    test_img = np.zeros((300,300,3))
    if np.array(input_image).shape[-1]==1:
        test_img[...,0]=input_image
        test_img[...,1]=input_image
        test_img[...,2]=input_image
        output_image = build_network(clicked.get(), test_img, visualize=True)
    else:
        output_image = build_network(clicked.get(), input_image, visualize=True)

    

def generate_GIF():
    pass

"""added data to get random image"""
def check_button():
    test_on_image()
    print('BUTTON PRESSED')


lab_test_type = Label(root, width=22, text="Testing medium: ", font = 'times 15 bold', anchor='w')
lab_test_type.place(relx=0.55, rely=0.7)
def update_test_type():
    global test_type
    test_type = var_test_type.get()
    print(var_test_type.get())
var_test_type = IntVar()
test_type = 1
R1 = Radiobutton(root, text="load_image", variable=var_test_type, value=1, bd = 2,  font = 'times 15', command = update_test_type)
R1.place(relx = 0.7, rely = 0.7)

R2 = Radiobutton(root, text="canvas_image", variable=var_test_type, value=2, bd = 2,  font = 'times 15', command = update_test_type)
R2.place(relx = 0.8, rely = 0.7) 



  # get image of the current location

def save_canvas():
    # canvas.postscript(file="file_name.ps")
    # img = Image.open("file_name" + '.ps') 
    image1.show()
    # image1.save("saving_img.jpg")
    print(image1) 
    print(np.asarray(image1))
    cv2.imwrite("cv2_write.jpg", np.asarray(image1))

    # im = ImageGrab.grab(rect)
    # im.save("grab.png")


# random_image = Button(root, text = 'Load Input Image',
#     command=check_button, bg= '#9F9FAF', bd=2)
# random_image.place(relx=0.7, rely=0.8)

predict = Button(root, text = 'GENERATE DESIGN',
    command=generate, bg= '#9B89B3', bd=2, font = 'times 24 bold')
predict.place(relx=0.35, rely=0.8)

# Position image

root.mainloop()