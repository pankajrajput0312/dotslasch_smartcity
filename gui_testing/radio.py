from tkinter import *

def sel():
   selection = "You selected the option " + str(var.get())
   label.config(text = selection)

# root = Tk()
# root.geometry("300x300")
# var = IntVar()
# R1 = Radiobutton(root, text="Option 1", variable=var, value=1,
#                   command=sel)
# R1.pack( anchor = W )

# R2 = Radiobutton(root, text="Option 2", variable=var, value=2,
#                   command=sel)
# R2.pack( anchor = W )

# R3 = Radiobutton(root, text="Option 3", variable=var, value=3,
#                   command=sel)
# R3.pack( anchor = W)

# label = Label(root)
# label.pack()

# var = IntVar()
# R1 = Radiobutton(root, text="Void", variable=var, value='void')
# R1.place(relx = 0.2, rely = 0.65)

# R2 = Radiobutton(root, text="Black", variable=var, value='black')
# R2.place(relx = 0.35, rely = 0.65)

# root.mainloop()


# ### testing 2
# from tkinter import *
# import tkinter.ttk as ttk


# root = Tk()                         # Main window
# myColor = '#40E0D0'                 # Its a light blue color
# root.configure(bg=myColor)          # Setting color of main window to myColor

# s = ttk.Style()                     # Creating style element
# s.configure('Wild.TRadiobutton',    # First argument is the name of style. Needs to end with: .TRadiobutton
#         background=myColor,         # Setting background to our specified color above
#         foreground='black')         # You can define colors like this also

# rb1 = ttk.Radiobutton(text = "works :)", style = 'Wild.TRadiobutton')       # Linking style with the button

# rb1.pack()                          # Placing Radiobutton

# root.mainloop()                     # Beginning loop


from tkinter import *
  
# Create object
root = Tk()
  
# Adjust size
root.geometry( "200x200" )
  
# Change the label text
def show():
    label.config( text = clicked.get() )
  
# Dropdown menu options
options = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]
  
# datatype of menu text
clicked = StringVar()
  
# initial menu text
clicked.set( "Monday" )
  
# Create Dropdown menu
drop = OptionMenu( root , clicked , *options )
drop.pack()
  
# Create button, it will change label text
button = Button( root , text = "click Me" , command = show ).pack()
  
# Create Label
label = Label( root , text = " " )
label.pack()
  
# Execute tkinter
root.mainloop()